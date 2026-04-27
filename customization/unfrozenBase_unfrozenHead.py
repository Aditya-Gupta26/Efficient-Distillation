"""
Experiment 4: Unfrozen distilled student (trainable copy)  +  Random DPT head (trained from scratch).

What is trained : EVERYTHING — the distilled student backbone AND the DPT head.
Student source  : checkpoints/best.pth  (in-memory copy only; file never modified)
DPT head        : RANDOM (Xavier) weights, trained from scratch

Key differences from other experiments:
  vs Exp 1 (frozenBase_pretrainedUnfrozenHead) : student is trainable AND head is random
  vs Exp 2 (unfrozenBase_pretrainedUnfrozenHead): head starts from random, not Intel weights
  vs Exp 3 (frozenBase_unfrozenHead)            : student is also trainable

This is the "full end-to-end" variant — everything is learned from the depth
task, with the distilled backbone providing a much better initialisation than
random weights would.

Uses StudentWithDPTUnfrozen (from shared.py): the forward() method has no
torch.no_grad() wrapper so gradients flow through the full model.

Checkpoints saved to: checkpoints/customization/unfrozenBase_unfrozenHead/

Usage:
    python customization/unfrozenBase_unfrozenHead.py \
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

from customization.shared import StudentWithDPTUnfrozen, run_training, get_fixed_val_indices
from data.nyu_depth_dataset import build_nyu_depth_dataloaders, NyuDepthDataset
from utils.device import get_device, pin_memory_for

EXPERIMENT  = "unfrozenBase_unfrozenHead"
DESCRIPTION = "Trainable distilled student + random DPT head trained from scratch (full end-to-end)"


def parse_args():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--student_checkpoint", default="checkpoints/best.pth")
    p.add_argument("--nyu_root",      default="data/nyu_depth_v2")
    p.add_argument("--save_dir",      default=f"checkpoints/customization/{EXPERIMENT}")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--lr_student",    type=float, default=1e-5,
                   help="LR for distilled student (lower — avoid forgetting ImageNet features)")
    p.add_argument("--lr_head",       type=float, default=1e-4,
                   help="LR for DPT head (higher — training from random init)")
    p.add_argument("--lr_min",        type=float, default=1e-6)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--amp",           action="store_true", default=True)
    p.add_argument("--no_amp",        action="store_false", dest="amp")
    p.add_argument("--wandb_project", default="efficient-distillation")
    p.add_argument("--wandb_entity",  default=None)
    return p.parse_args()


def build_model(student_checkpoint: str) -> StudentWithDPTUnfrozen:
    """
    Build the model for this experiment:
      - student backbone : loaded from checkpoint, TRAINABLE (in-memory copy)
      - DPT head         : RANDOM Xavier init (pretrained=False), TRAINABLE
      - forward()        : no torch.no_grad() (StudentWithDPTUnfrozen)
    """
    model = StudentWithDPTUnfrozen(
        student_checkpoint = student_checkpoint,
        dpt_pretrained     = False,   # ← random head — no Intel pretrained weights
        freeze_student     = False,   # ← student backbone is trainable
    )
    student_params = sum(p.numel() for p in model.student.parameters())
    head_params    = sum(p.numel() for p in model.dpt_head.parameters())
    print(f"[{EXPERIMENT}] student={student_params:,}  head={head_params:,}  "
          f"(both trainable from random head init)", flush=True)
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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.student_checkpoint).to(device)

    # ── Optimiser — two param groups (lower LR for student) ──────────────────
    optimiser = AdamW([
        {"params": model.student.parameters(), "lr": args.lr_student},
        {"params": model.dpt_head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr_min)

    args.lr = args.lr_head   # used by shared run_training for LR logging

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
            tags    = ["unfrozen-student", "random-head"],
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
