"""
Experiment 2: Unfrozen distilled student (trainable copy)  +  Pretrained DPT head (trainable).

What is trained : EVERYTHING — both the distilled student backbone AND the DPT head.
Student source  : checkpoints/best.pth  (loaded into memory; original file is NEVER modified)
DPT head        : pretrained Intel/dpt-swinv2-tiny-256 weights, trainable

Key difference from Experiment 1 (frozenBase_pretrainedUnfrozenHead):
  The student backbone is NOT frozen. Gradients flow through the entire model,
  allowing end-to-end fine-tuning on the depth task.

  We use StudentWithDPTUnfrozen (from shared.py) which overrides forward() to
  remove the torch.no_grad() wrapper so gradients reach the student.

  Two separate learning rates:
    - Student backbone: lr_student (default 1e-5, lower — already well-trained)
    - DPT head        : lr_head    (default 1e-4, higher — needs more adaptation)

Checkpoints saved to: checkpoints/customization/unfrozenBase_pretrainedUnfrozenHead/

Usage:
    python customization/unfrozenBase_pretrainedUnfrozenHead.py \
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

from customization.shared import (
    StudentWithDPTUnfrozen,   # ← key: allows gradients through student
    run_training,
    get_fixed_val_indices,
)
from data.nyu_depth_dataset import build_nyu_depth_dataloaders, NyuDepthDataset
from utils.device import get_device, pin_memory_for, get_distributed_info
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_ddp,
    is_main_process,
    barrier,
)

EXPERIMENT  = "unfrozenBase_pretrainedUnfrozenHead"
DESCRIPTION = "Trainable distilled student + pretrained Intel DPT head (both trainable)"


def parse_args():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--student_checkpoint", default="checkpoints/best.pth")
    p.add_argument("--nyu_root",      default="data/nyu_depth_v2")
    p.add_argument("--save_dir",      default=f"checkpoints/customization/{EXPERIMENT}")
    p.add_argument("--epochs",        type=int,   default=30)
    # Two LRs: lower for the student (already trained), higher for the DPT head
    p.add_argument("--lr_student",    type=float, default=1e-5,
                   help="LR for distilled student backbone (lower — already well-trained)")
    p.add_argument("--lr_head",       type=float, default=1e-4,
                   help="LR for DPT head (higher — needs more adaptation)")
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
      - student backbone : loaded from checkpoint, TRAINABLE (freeze_student=False)
      - DPT head         : pretrained Intel/dpt-swinv2-tiny-256 weights, TRAINABLE
      - forward()        : no torch.no_grad() wrapper (StudentWithDPTUnfrozen)
    """
    model = StudentWithDPTUnfrozen(
        student_checkpoint = student_checkpoint,
        dpt_pretrained     = True,    # ← pretrained Intel DPT head weights
        freeze_student     = False,   # ← student backbone is trainable
    )
    student_params = sum(p.numel() for p in model.student.parameters())
    head_params    = sum(p.numel() for p in model.dpt_head.parameters())
    print(f"[{EXPERIMENT}] student={student_params:,}  head={head_params:,}  "
          f"(both trainable)", flush=True)
    return model


def main():
    args = parse_args()
    dist_info = setup_distributed()
    device = dist_info["device"]
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"  {EXPERIMENT}")
        print(f"  {DESCRIPTION}")
        print(f"  Device: {device}  |  Epochs: {args.epochs}")
        print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_nyu_depth_dataloaders(
        root=args.nyu_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory_for(device),
        distributed=dist_info["distributed"],
        rank=dist_info["rank"],
        world_size=dist_info["world_size"],
    )
    val_dataset = NyuDepthDataset(root=args.nyu_root, split="val", img_size=args.img_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.student_checkpoint).to(device)
    model = wrap_ddp(model, device)

    # ── Optimiser — two param groups with different learning rates ────────────
    # The student is already well-trained so it gets a much lower LR to avoid
    # catastrophic forgetting of its ImageNet-trained representations.
    base_model = model.module if hasattr(model, "module") else model
    optimiser = AdamW([
        {"params": base_model.student.parameters(), "lr": args.lr_student},
        {"params": base_model.dpt_head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    # Use the higher lr (head) as T_max reference for the scheduler
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr_min)

    # Store lr as the head lr for shared run_training logging
    args.lr = args.lr_head

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = None
    if is_main_process():
        try:
            import wandb
            wandb_run = wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                name    = EXPERIMENT,
                config  = vars(args),
                resume  = "allow",
                tags    = ["unfrozen-student", "pretrained-head"],
            )
            print(f"[wandb] {wandb_run.url}\n")
        except Exception as e:
            print(f"[wandb] Disabled ({e})\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        run_training(
            model=model, train_loader=train_loader, val_loader=val_loader,
            val_dataset=val_dataset, optimiser=optimiser, scheduler=scheduler,
            device=device, args=args, run_name=EXPERIMENT, wandb_run=wandb_run,
        )
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
