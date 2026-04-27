"""
Experiment 1: Frozen distilled student  +  Pretrained DPT head (trainable).

What is frozen  : The distilled student backbone (checkpoints/best.pth).
                  Its weights never change — it is used purely as a feature extractor.
What is trained : The DPT neck + head, initialised from Intel/dpt-swinv2-tiny-256
                  pretrained weights.

Hypothesis: The pretrained head already knows how to decode DPT features.
Even though it was trained on SwinV2-Tiny (our backbone is Swin-Tiny v1),
the InstanceNorm in DPTDepthHead normalises the feature statistics so the
pretrained weights provide a better starting point than random init.

Checkpoints saved to: checkpoints/customization/frozenBase_pretrainedUnfrozenHead/

Usage:
    python customization/frozenBase_pretrainedUnfrozenHead.py \
        --student_checkpoint checkpoints/best.pth \
        --nyu_root data/nyu_depth_v2 \
        --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Shared utilities — metrics, training loop, checkpoint helpers, W&B preview
from customization.shared import (
    run_training, validate,
    get_fixed_val_indices,
)
from models.depth_model import StudentWithDPT          # frozen-student model
from data.nyu_depth_dataset import build_nyu_depth_dataloaders, NyuDepthDataset
from utils.device import get_device, pin_memory_for

# ── Experiment identity ────────────────────────────────────────────────────────
EXPERIMENT  = "frozenBase_pretrainedUnfrozenHead"
DESCRIPTION = "Frozen distilled student + pretrained Intel DPT head (trainable)"


def parse_args():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--student_checkpoint", default="checkpoints/best.pth",
                   help="Distilled student backbone — ALWAYS required, no fallback")
    p.add_argument("--nyu_root",     default="data/nyu_depth_v2")
    p.add_argument("--save_dir",     default=f"checkpoints/customization/{EXPERIMENT}")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Learning rate for the DPT head (student is frozen)")
    p.add_argument("--lr_min",       type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--img_size",     type=int,   default=224)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--no_amp",       action="store_false", dest="amp")
    p.add_argument("--wandb_project",default="efficient-distillation")
    p.add_argument("--wandb_entity", default=None)
    return p.parse_args()


def build_model(student_checkpoint: str) -> StudentWithDPT:
    """
    Build the model for this experiment:
      - student backbone : loaded from checkpoint, ALL WEIGHTS FROZEN
      - DPT head         : pretrained Intel/dpt-swinv2-tiny-256 weights, TRAINABLE
    """
    model = StudentWithDPT(
        student_checkpoint = student_checkpoint,
        dpt_pretrained     = True,   # ← pretrained Intel DPT head weights
        freeze_student     = True,   # ← student backbone is frozen
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[{EXPERIMENT}] total={total:,}  trainable={trainable:,}  "
          f"(DPT head only — student frozen)", flush=True)
    return model


def main():
    args   = parse_args()
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  {EXPERIMENT}")
    print(f"  {DESCRIPTION}")
    print(f"  Device: {device}  |  Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_nyu_depth_dataloaders(
        root=args.nyu_root, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=pin_memory_for(device),
    )
    val_dataset = NyuDepthDataset(root=args.nyu_root, split="val", img_size=args.img_size)
    print(f"Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.student_checkpoint).to(device)

    # ── Optimiser — only DPT head parameters (student is frozen) ─────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr_min)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = EXPERIMENT,
            config  = vars(args),
            resume  = "allow",
            tags    = ["frozen-student", "pretrained-head"],
        )
        print(f"[wandb] {wandb_run.url}\n")
    except Exception as e:
        print(f"[wandb] Disabled ({e})\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    run_training(
        model=model, train_loader=train_loader, val_loader=val_loader,
        val_dataset=val_dataset, optimiser=optimiser, scheduler=scheduler,
        device=device, args=args, run_name=EXPERIMENT, wandb_run=wandb_run,
    )


if __name__ == "__main__":
    main()
