"""
Distillation Trainer.

Orchestrates the full training loop:
    1. Forward pass through the frozen/partially-frozen teacher.
    2. Forward pass through the student.
    3. Adapter projection of student features.
    4. Compute combined distillation + task loss.
    5. Backward pass and optimiser step.
    6. LR-scheduler step.
    7. Logging & checkpointing.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.teacher import SwinTeacher
from models.student import SwinStudentTiny
from models.adapters import FeatureAdapter
from distillation.losses import DistillationLoss
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import compute_metrics


class DistillationTrainer:
    """
    Full distillation training loop.

    Args:
        teacher       : Pretrained :class:`SwinTeacher`.
        student       : :class:`SwinStudentTiny` to be trained.
        adapter       : :class:`FeatureAdapter` bridging the two backbones.
        loss_fn       : :class:`DistillationLoss` instance.
        train_loader  : DataLoader for COCO training split.
        val_loader    : DataLoader for COCO validation split.
        cfg           : Flat config dict (see ``configs/distill_config.yaml``).
        device        : torch.device.
    """

    def __init__(
        self,
        teacher: SwinTeacher,
        student: SwinStudentTiny,
        adapter: FeatureAdapter,
        loss_fn: DistillationLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ):
        self.teacher      = teacher.to(device)
        self.student      = student.to(device)
        self.adapter      = adapter.to(device)
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.logger       = setup_logger("DistillationTrainer")

        # Teacher is always in eval mode — we only distil from it
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Optimiser: only student + adapter parameters
        trainable_params = list(self.student.parameters()) + list(self.adapter.parameters())
        self.optimiser = torch.optim.AdamW(
            trainable_params,
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 1e-2),
        )

        # Cosine LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser,
            T_max=cfg.get("epochs", 30),
            eta_min=cfg.get("lr_min", 1e-6),
        )

        # Mixed-precision scaler
        self.scaler = GradScaler(enabled=cfg.get("amp", True))

        self.start_epoch = 0
        self.best_metric = 0.0

        # Resume if checkpoint exists
        ckpt_path = cfg.get("resume", None)
        if ckpt_path and os.path.isfile(ckpt_path):
            self.start_epoch, self.best_metric = load_checkpoint(
                ckpt_path, self.student, self.adapter, self.optimiser, self.scheduler
            )
            self.logger.info(f"Resumed from checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full training loop."""
        epochs = self.cfg.get("epochs", 30)
        save_dir = self.cfg.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(self.start_epoch, epochs):
            train_losses = self._train_one_epoch(epoch)
            val_metrics  = self._validate(epoch)

            self.scheduler.step()

            # Log epoch summary
            self.logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {train_losses['total']:.4f} "
                f"(feat={train_losses['feat']:.4f}, "
                f"at={train_losses['at']:.4f}, "
                f"kd={train_losses['kd']:.4f}) | "
                f"Val mAP: {val_metrics.get('mAP', 0.0):.4f}"
            )

            # Save best checkpoint
            current_metric = val_metrics.get("mAP", 0.0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric

            save_checkpoint(
                path=os.path.join(save_dir, f"epoch_{epoch+1:03d}.pth"),
                epoch=epoch + 1,
                student=self.student,
                adapter=self.adapter,
                optimiser=self.optimiser,
                scheduler=self.scheduler,
                best_metric=self.best_metric,
                is_best=is_best,
            )

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Single training epoch."""
        self.student.train()
        self.adapter.train()
        self.teacher.eval()

        running = {k: 0.0 for k in ("total", "feat", "at", "kd", "task")}
        n_batches = len(self.train_loader)
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)

            self.optimiser.zero_grad()

            with autocast(enabled=self.cfg.get("amp", True)):
                # Teacher forward (no_grad – already set via requires_grad=False)
                with torch.no_grad():
                    t_feats, t_logits = self.teacher(images)

                # Student forward
                s_feats, s_logits = self.student(images)

                # Adapter projection
                adapted_s_feats = self.adapter(s_feats, t_feats)

                # Compute losses
                loss_dict = self.loss_fn(
                    adapted_student_feats=adapted_s_feats,
                    teacher_feats=t_feats,
                    student_feats_raw=s_feats,
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                )

            self.scaler.scale(loss_dict["total"]).backward()
            self.scaler.unscale_(self.optimiser)
            nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.adapter.parameters()),
                max_norm=self.cfg.get("grad_clip", 1.0),
            )
            self.scaler.step(self.optimiser)
            self.scaler.update()

            for k in running:
                running[k] += loss_dict[k].item()

            if batch_idx % self.cfg.get("log_every", 50) == 0:
                elapsed = time.time() - t0
                self.logger.info(
                    f"  [Epoch {epoch+1} | {batch_idx}/{n_batches}] "
                    f"loss={loss_dict['total'].item():.4f}  "
                    f"({elapsed:.1f}s elapsed)"
                )

        return {k: v / n_batches for k, v in running.items()}

    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validation pass – returns a dict of metrics."""
        self.student.eval()
        self.adapter.eval()

        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device)
                targets = batch.get("targets", None)

                s_feats, s_logits = self.student(images)

                if s_logits is not None:
                    all_preds.append(s_logits.cpu())
                if targets is not None:
                    all_targets.append(targets.cpu())

        if all_preds and all_targets:
            preds   = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            metrics = compute_metrics(preds, targets)
        else:
            metrics = {}

        return metrics
