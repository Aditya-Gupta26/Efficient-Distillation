"""
Main entry-point for knowledge distillation training.

Usage:
    python train.py --config configs/distill_config.yaml
    python train.py --config configs/distill_config.yaml --resume checkpoints/epoch_010.pth
"""

from __future__ import annotations

import argparse
import yaml
import torch

from models.teacher import SwinTeacher
from models.student import SwinStudentTiny
from models.adapters import FeatureAdapter
from distillation.losses import DistillationLoss
from distillation.trainer import DistillationTrainer
from data.imagenet_dataset import build_imagenet_dataloaders
from utils.logger import setup_logger
from utils.wandb_logger import init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Swin-Transformer Knowledge Distillation")
    parser.add_argument("--config", type=str, default="configs/distill_config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from.")
    parser.add_argument("--adapter_warm_start", type=str, default=None,
                        help="Path to a checkpoint whose adapter weights are used to "
                             "warm-start the adapters only (student & optimiser stay fresh). "
                             "Ignored when --resume is also provided.")
    parser.add_argument("--override", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="Override config values, e.g. --override w_feat=0.1 temperature=6.0 feat_loss_type=mse")
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply KEY=VALUE overrides to a config dict, casting to int/float/bool where appropriate."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE format.")
        key, raw = item.split("=", 1)
        # Type casting: try int → float → bool → string
        for cast in (int, float):
            try:
                raw = cast(raw); break
            except ValueError:
                pass
        else:
            if raw.lower() in ("true", "false"):
                raw = raw.lower() == "true"
        cfg[key] = raw
    return cfg


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    cfg    = apply_overrides(cfg, args.override)
    logger = setup_logger("train")

    # Allow CLI override for resume path
    if args.resume:
        cfg["resume"] = args.resume

    # Allow CLI override for adapter warm-start path (ignored if --resume is also set)
    if args.adapter_warm_start and not cfg.get("resume"):
        cfg["adapter_warm_start"] = args.adapter_warm_start

    # ------------------------------------------------------------------ #
    # Weights & Biases
    # ------------------------------------------------------------------ #
    # Smoke-test: skip wandb entirely
    if cfg.get("smoke_test_batches", 0):
        cfg.setdefault("wandb", {})["enabled"] = False
        logger.info("smoke_test_batches set — W&B disabled for this run")

    # Build a descriptive run name from any CLI overrides
    if args.override and cfg.get("wandb", {}).get("run_name") is None:
        tag = "_".join(o.replace("=", "") for o in args.override)
        cfg.setdefault("wandb", {})["run_name"] = tag

    wandb_run = init_wandb(cfg)

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    train_loader, val_loader = build_imagenet_dataloaders(
        imagenet_root = cfg["imagenet_root"],
        img_size      = cfg.get("img_size", 224),
        batch_size    = cfg.get("batch_size", 32),
        num_workers   = cfg.get("num_workers", 8),
        pin_memory    = cfg.get("pin_memory", True),
    )
    logger.info(
        f"ImageNet  train: {len(train_loader.dataset):,} images  |  "
        f"val: {len(val_loader.dataset):,} images"
    )

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    teacher_variant = cfg.get("teacher_variant", "swin_large")
    teacher = SwinTeacher(
        variant       = teacher_variant,
        pretrained    = cfg.get("teacher_pretrained", True),
        num_classes   = cfg.get("num_classes", 1000),
        frozen_stages = cfg.get("teacher_frozen_stages", 4),   # fully frozen
    )
    student = SwinStudentTiny(
        pretrained  = cfg.get("student_pretrained", True),
        num_classes = cfg.get("num_classes", 1000),
    )
    logger.info(
        f"Teacher ({teacher_variant})  params: {teacher.num_parameters:,}  "
        f"trainable: {teacher.num_trainable_parameters:,}"
    )
    logger.info(
        f"Student (swin_tiny)    params: {student.num_parameters:,}  "
        f"trainable: {student.num_trainable_parameters:,}"
    )

    # ------------------------------------------------------------------ #
    # Adapters
    # ------------------------------------------------------------------ #
    adapter = FeatureAdapter(
        student_channels = student.stage_channels,
        teacher_channels = teacher.stage_channels,
        stages           = cfg.get("adapter_stages", [0, 1, 2, 3]),
        use_spatial_align= cfg.get("adapter_spatial_align", True),
    )
    logger.info(f"Adapter params: {adapter.num_parameters:,}")

    # ------------------------------------------------------------------ #
    # torch.compile  (~15-30% throughput gain on A100, free)
    # Requires triton; falls back to eager if not available.
    # ------------------------------------------------------------------ #
    if cfg.get("compile", True) and hasattr(torch, "compile"):
        try:
            import triton  # noqa: F401
            logger.info("Compiling student and adapter with torch.compile ...")
            student = torch.compile(student)
            adapter = torch.compile(adapter)
            logger.info("torch.compile done.")
        except ImportError:
            logger.warning("triton not found — skipping torch.compile (pip install triton to enable).")

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #
    loss_fn = DistillationLoss(
        w_feat         = cfg.get("w_feat", 1.0),
        w_at           = cfg.get("w_at",   0.5),
        w_kd           = cfg.get("w_kd",   1.0),
        w_task         = cfg.get("w_task",  1.0),
        temperature    = cfg.get("temperature", 4.0),
        feat_loss_type = cfg.get("feat_loss_type", "mse"),
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    trainer = DistillationTrainer(
        teacher      = teacher,
        student      = student,
        adapter      = adapter,
        loss_fn      = loss_fn,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        device       = device,
        wandb_run    = wandb_run,
    )
    trainer.train()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
