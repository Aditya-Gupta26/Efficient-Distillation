"""
DepthTrainer: trains the DPT depth head on top of the frozen distilled student.

Only the DPT head (neck + prediction layers) is optimised.
The student backbone stays frozen and in eval mode throughout.

Loss:
    Scale-Invariant Log loss (SILog) — the standard for monocular depth.
    L = sqrt(Var(d) + lambda * Mean(d)^2)   where d = log(pred) - log(gt)
    lambda=0.85 follows AdaBins / DPT convention.

Metrics (reported on validation):
    RMSE     root mean squared error (metres)
    AbsRel   mean absolute relative error
    delta1   fraction of pixels with max(pred/gt, gt/pred) < 1.25
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from PIL import Image

from utils.logger import setup_logger

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_VIS_N_IMAGES  = 4   # number of val images to visualise per checkpoint


def _denorm_rgb(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor → (H,W,3) uint8."""
    img = (t.cpu() * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _depth_to_colormap(arr: np.ndarray, mask_zeros: bool = False) -> np.ndarray:
    """Float depth array → (H,W,3) uint8 plasma colormap."""
    if mask_zeros:
        valid = arr[arr > 0]
        lo, hi = (valid.min(), valid.max()) if valid.size else (0.0, 1.0)
    else:
        lo, hi = arr.min(), arr.max()
    norm = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        rgba = cm.plasma(gray.astype(np.float32) / 255.0)
        return (rgba[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        return np.stack([gray, gray, gray], axis=-1)


class DepthTrainer:
    """
    Trains the DPT depth head (models.depth_model.StudentWithDPT).

    Args:
        model:        StudentWithDPT instance (student frozen, DPT head trainable).
        train_loader: DataLoader yielding {"images": ..., "depths": ...}.
        val_loader:   same format, no augmentation.
        cfg:          config dict (from depth_config.yaml).
        device:       torch.device.
        wandb_run:    optional W&B run object.
    """

    SILOG_LAMBDA = 0.5   # 0.85 is nearly scale-invariant; 0.5 gives stronger scale gradient

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          dict,
        device:       torch.device,
        wandb_run=None,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.wandb_run    = wandb_run
        self.logger       = setup_logger("depth_trainer")

        # Only optimise DPT head parameters
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

        self.optimiser = AdamW(
            trainable,
            lr           = cfg.get("lr", 1e-4),
            weight_decay = cfg.get("weight_decay", 1e-2),
        )
        warmup_epochs = cfg.get("warmup_epochs", 5)
        cosine = CosineAnnealingLR(
            self.optimiser,
            T_max   = max(cfg.get("epochs", 50) - warmup_epochs, 1),
            eta_min = cfg.get("lr_min", 1e-6),
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimiser,
            start_factor = 0.1,
            end_factor   = 1.0,
            total_iters  = warmup_epochs,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimiser,
            schedulers  = [warmup, cosine],
            milestones  = [warmup_epochs],
        )
        self.scaler = GradScaler("cuda", enabled=cfg.get("amp", True))

        self.epochs      = cfg.get("epochs", 50)
        self.log_every   = cfg.get("log_every", 20)
        self.save_dir    = Path(cfg.get("save_dir", "checkpoints/depth"))
        self.grad_clip   = cfg.get("grad_clip", 1.0)
        self.amp         = cfg.get("amp", True)

        self.start_epoch  = 0
        self.best_rmse    = float("inf")

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.save_dir / "vis"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Fix a small val batch for consistent epoch visualisations
        self._vis_batch = self._grab_vis_batch()

        if cfg.get("resume"):
            self._resume(cfg["resume"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_metrics = self._train_one_epoch(epoch)
            val_metrics   = self._validate(epoch)
            self.scheduler.step()

            elapsed = time.time() - t0
            self.logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs}  "
                f"loss={train_metrics['loss']:.4f}  "
                f"rmse={val_metrics['rmse']:.4f}  "
                f"abs_rel={val_metrics['abs_rel']:.4f}  "
                f"delta1={val_metrics['delta1']:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                f"time={elapsed:.1f}s"
            )

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch":          epoch + 1,
                    "train/loss":     train_metrics["loss"],
                    "val/rmse":       val_metrics["rmse"],
                    "val/abs_rel":    val_metrics["abs_rel"],
                    "val/delta1":     val_metrics["delta1"],
                    "lr":             self.scheduler.get_last_lr()[0],
                })

            # Auto-rollback if RMSE spikes badly (scale explosion)
            if val_metrics["rmse"] > self.best_rmse * 3.0 and self.best_rmse < float("inf"):
                self.logger.warning(
                    f"  RMSE {val_metrics['rmse']:.2f} >> 3× best {self.best_rmse:.2f} "
                    f"— rolling back to best checkpoint and halving LR"
                )
                self._rollback_to_best()
            else:
                self._save(epoch, val_metrics["rmse"])

            if (epoch + 1) % self.cfg.get("vis_every", 10) == 0:
                self._visualize(epoch + 1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        # Ensure student backbone stays frozen+eval
        if hasattr(self.model, "student"):
            self.model.student.eval()

        total_loss = 0.0
        n_batches  = len(self.train_loader)

        for i, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device, non_blocking=True)
            depths = batch["depths"].to(self.device, non_blocking=True)

            self.optimiser.zero_grad()

            with autocast("cuda", enabled=self.amp):
                pred = self.model(images)          # (B, 1, H, W)
                pred = pred.squeeze(1)             # (B, H, W)
                loss = self._silog_loss(pred, depths)

            self.scaler.scale(loss).backward()

            if self.grad_clip:
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )

            self.scaler.step(self.optimiser)
            self.scaler.update()

            total_loss += loss.item()

            if (i + 1) % self.log_every == 0:
                self.logger.info(
                    f"  [{epoch+1}][{i+1}/{n_batches}]  loss={loss.item():.4f}"
                )

        return {"loss": total_loss / n_batches}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()

        rmse_sum    = 0.0
        absrel_sum  = 0.0
        delta1_sum  = 0.0
        n = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device, non_blocking=True)
            depths = batch["depths"].to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.amp):
                pred = self.model(images).squeeze(1)  # (B, H, W)

            m = self._depth_metrics(pred, depths)
            b = images.size(0)

            rmse_sum   += m["rmse"]   * b
            absrel_sum += m["abs_rel"] * b
            delta1_sum += m["delta1"] * b
            n          += b

        return {
            "rmse":    rmse_sum   / n,
            "abs_rel": absrel_sum / n,
            "delta1":  delta1_sum / n,
        }

    # ------------------------------------------------------------------
    # Loss & metrics
    # ------------------------------------------------------------------

    def _silog_loss(
        self,
        pred:   torch.Tensor,   # (B, H, W)
        target: torch.Tensor,   # (B, H, W)
    ) -> torch.Tensor:
        """Scale-Invariant Log loss."""
        mask = (target > 0) & (pred > 1e-6)
        if mask.sum() == 0:
            return pred.sum() * 0.0  # zero loss, keeps graph alive

        d = torch.log(pred[mask]) - torch.log(target[mask])
        variance = d.var()
        mean_sq  = d.mean() ** 2
        return torch.sqrt(variance + (1.0 - self.SILOG_LAMBDA) * mean_sq + 1e-8)

    @staticmethod
    def _depth_metrics(
        pred:   torch.Tensor,   # (B, H, W)
        target: torch.Tensor,   # (B, H, W)
    ) -> dict:
        mask = (target > 0) & (pred > 1e-6)
        if mask.sum() == 0:
            return {"rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}

        p = pred[mask]
        t = target[mask]

        rmse    = torch.sqrt(((p - t) ** 2).mean()).item()
        abs_rel = (torch.abs(p - t) / t).mean().item()
        ratio   = torch.max(p / t, t / p)
        delta1  = (ratio < 1.25).float().mean().item()

        return {"rmse": rmse, "abs_rel": abs_rel, "delta1": delta1}

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _grab_vis_batch(self) -> dict:
        """Pull the first _VIS_N_IMAGES samples from the val loader and pin them."""
        images, depths = [], []
        for batch in self.val_loader:
            images.append(batch["images"])
            depths.append(batch["depths"])
            if sum(t.size(0) for t in images) >= _VIS_N_IMAGES:
                break
        images = torch.cat(images, dim=0)[:_VIS_N_IMAGES]
        depths = torch.cat(depths, dim=0)[:_VIS_N_IMAGES]
        return {"images": images, "depths": depths}

    @torch.no_grad()
    def _visualize(self, epoch: int) -> None:
        self.model.eval()
        images = self._vis_batch["images"].to(self.device)
        depths = self._vis_batch["depths"]          # keep on CPU for numpy

        with autocast("cuda", enabled=self.amp):
            preds = self.model(images).squeeze(1).cpu()   # (N, H, W)

        border = np.ones((images.shape[2], 4, 3), dtype=np.uint8) * 180
        wandb_images = []

        for i in range(images.size(0)):
            rgb_panel  = _denorm_rgb(images[i].cpu())
            pred_panel = _depth_to_colormap(preds[i].numpy(),  mask_zeros=False)
            gt_panel   = _depth_to_colormap(depths[i].numpy(), mask_zeros=True)

            row = np.concatenate([rgb_panel, border, pred_panel, border, gt_panel], axis=1)

            # Save to disk
            out_path = self.vis_dir / f"epoch_{epoch:03d}_sample_{i:02d}.png"
            Image.fromarray(row).save(out_path)

            if self.wandb_run is not None:
                import wandb
                wandb_images.append(
                    wandb.Image(row, caption=f"epoch {epoch} | sample {i} | RGB / Pred / GT")
                )

        if self.wandb_run is not None:
            self.wandb_run.log({"val/depth_vis": wandb_images, "epoch": epoch})
            self.logger.info(f"  Uploaded {len(wandb_images)} depth visualisations to W&B (epoch {epoch})")
        else:
            self.logger.info(f"  Saved {images.size(0)} depth visualisations to {self.vis_dir}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save(self, epoch: int, rmse: float) -> None:
        state = {
            "epoch":      epoch + 1,
            "model":      self.model.state_dict(),
            "optimiser":  self.optimiser.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "best_rmse":  self.best_rmse,
        }
        path = self.save_dir / f"epoch_{epoch+1:03d}.pth"
        torch.save(state, path)

        if rmse < self.best_rmse:
            self.best_rmse = rmse
            best_path = self.save_dir / "best.pth"
            torch.save(state, best_path)
            self.logger.info(f"  ✓ New best RMSE {rmse:.4f} — saved {best_path}")

    def _rollback_to_best(self) -> None:
        best_path = self.save_dir / "best.pth"
        if not best_path.exists():
            self.logger.warning("  No best.pth found — cannot roll back.")
            return
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        # Halve LR for all param groups
        for pg in self.optimiser.param_groups:
            pg["lr"] *= 0.5
        self.logger.info(
            f"  Rolled back to epoch {ckpt['epoch']}  "
            f"(best RMSE {self.best_rmse:.4f})  "
            f"new LR={self.optimiser.param_groups[0]['lr']:.2e}"
        )

    def _resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"]
        self.best_rmse   = ckpt.get("best_rmse", float("inf"))
        self.logger.info(f"Resumed from {path} (epoch {self.start_epoch})")
