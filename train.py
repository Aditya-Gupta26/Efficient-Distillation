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
from data.coco_dataset import build_coco_dataloaders
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Swin-Transformer Knowledge Distillation")
    parser.add_argument("--config", type=str, default="configs/distill_config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    logger = setup_logger("train")

    # Allow CLI override for resume path
    if args.resume:
        cfg["resume"] = args.resume

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
    train_loader, val_loader = build_coco_dataloaders(
        coco_root   = cfg["coco_root"],
        img_size    = cfg.get("img_size", 224),
        batch_size  = cfg.get("batch_size", 32),
        num_workers = cfg.get("num_workers", 8),
        pin_memory  = cfg.get("pin_memory", True),
    )
    logger.info(
        f"COCO  train: {len(train_loader.dataset):,} images  |  "
        f"val: {len(val_loader.dataset):,} images"
    )

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    teacher_variant = cfg.get("teacher_variant", "swin_large")
    teacher = SwinTeacher(
        variant       = teacher_variant,
        pretrained    = cfg.get("teacher_pretrained", True),
        num_classes   = cfg.get("num_classes", 80),
        frozen_stages = cfg.get("teacher_frozen_stages", 4),   # fully frozen
    )
    student = SwinStudentTiny(
        pretrained  = cfg.get("student_pretrained", True),
        num_classes = cfg.get("num_classes", 80),
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
    )
    trainer.train()


if __name__ == "__main__":
    main()
