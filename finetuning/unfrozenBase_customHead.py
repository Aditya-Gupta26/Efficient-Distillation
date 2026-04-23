"""
Full fine-tuning: both student backbone and DPT head train together.

Loads a depth checkpoint and unfreezes the Swin-Tiny backbone so that both
the distilled student weights and the DPT head are updated.  A lower LR is
used for the backbone to avoid destroying the distilled representations.

Usage:
    python finetuning/unfrozenBase_customHead.py \
        --student_checkpoint checkpoints/epoch_069.pth \
        --depth_checkpoint   checkpoints/depth_head/best.pth \
        --nyu_root           data/nyu_depth_v2 \
        --epochs             20 \
        --save_dir           checkpoints/finetune_unfrozen
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.depth_model import StudentWithDPT
from data.nyu_depth_dataset import build_nyu_depth_dataloaders
from utils.device import get_device, maybe_autocast, make_scaler, pin_memory_for


# ─────────────────────────────────────────────────────────────────────────────
# Loss & metrics
# ─────────────────────────────────────────────────────────────────────────────

SILOG_LAMBDA = 0.85


def silog_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = (target > 0) & (pred > 1e-6)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    d = torch.log(pred[mask]) - torch.log(target[mask])
    return torch.sqrt(d.var() + (1.0 - SILOG_LAMBDA) * d.mean() ** 2 + 1e-8)


@torch.no_grad()
def depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mask = (target > 0) & (pred > 1e-6)
    if mask.sum() == 0:
        return {"rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}
    p, t = pred[mask], target[mask]
    rmse    = torch.sqrt(((p - t) ** 2).mean()).item()
    abs_rel = (torch.abs(p - t) / t).mean().item()
    delta1  = (torch.max(p / t, t / p) < 1.25).float().mean().item()
    return {"rmse": rmse, "abs_rel": abs_rel, "delta1": delta1}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(student_checkpoint: str, depth_checkpoint: str) -> StudentWithDPT:
    model = StudentWithDPT(
        student_checkpoint=student_checkpoint,
        dpt_pretrained=False,
        freeze_student=False,
    )

    ckpt  = torch.load(depth_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] Missing keys  : {len(missing)}  (e.g. {missing[:3]})")
    if unexpected:
        print(f"[load] Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    for p in model.parameters():
        p.requires_grad = True

    backbone = sum(p.numel() for p in model.student.parameters())
    head     = sum(p.numel() for p in model.dpt_head.parameters())
    print(
        f"Model loaded (fully unfrozen)\n"
        f"  Backbone (student): {backbone:,}  |  DPT head: {head:,}  |  Total: {backbone+head:,}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, scaler, device, grad_clip, amp, epoch):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)
    t0         = time.time()

    for i, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        depths = batch["depths"].to(device, non_blocking=True)

        optimiser.zero_grad()
        with maybe_autocast(device, amp):
            pred = model(images).squeeze(1)
            loss = silog_loss(pred, depths)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item()

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(
                f"  epoch {epoch+1}  [{i+1:>3}/{n_batches}]  "
                f"loss={loss.item():.4f}  ({time.time()-t0:.0f}s)",
                end="\r", flush=True,
            )

    print()
    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    rmse_sum = abs_rel_sum = delta1_sum = 0.0
    n = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        depths = batch["depths"].to(device, non_blocking=True)
        with maybe_autocast(device, amp):
            pred = model(images).squeeze(1)
        m = depth_metrics(pred, depths)
        b = images.size(0)
        rmse_sum    += m["rmse"]    * b
        abs_rel_sum += m["abs_rel"] * b
        delta1_sum  += m["delta1"]  * b
        n           += b
    return {"rmse": rmse_sum / n, "abs_rel": abs_rel_sum / n, "delta1": delta1_sum / n}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full fine-tuning: student backbone + DPT head")
    p.add_argument("--student_checkpoint", required=True)
    p.add_argument("--depth_checkpoint",   required=True,
                   help="Depth head checkpoint to start from (e.g. checkpoints/depth_head/best.pth)")
    p.add_argument("--nyu_root",      required=True)
    p.add_argument("--save_dir",      default="checkpoints/finetune_unfrozen")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--lr_backbone",   type=float, default=1e-5)
    p.add_argument("--lr_head",       type=float, default=5e-5)
    p.add_argument("--lr_min",        type=float, default=1e-7)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--amp",           action="store_true", default=True)
    p.add_argument("--no_amp",        action="store_false", dest="amp")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_nyu_depth_dataloaders(
        root=args.nyu_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory_for(device),
    )
    print(f"NYU-Depth  train: {len(train_loader.dataset):,}  |  val: {len(val_loader.dataset):,}")

    model = load_model(args.student_checkpoint, args.depth_checkpoint).to(device)

    optimiser = AdamW(
        [
            {"params": model.student.parameters(),  "lr": args.lr_backbone},
            {"params": model.dpt_head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr_min)
    scaler    = make_scaler(device, args.amp)
    best_rmse = float("inf")

    print(f"\nLR backbone={args.lr_backbone:.1e}  head={args.lr_head:.1e}")
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val RMSE':>9}  {'AbsRel':>8}  {'δ1':>7}  {'LR_head':>9}  Time")
    print("─" * 74)

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimiser, scaler, device, args.grad_clip, args.amp, epoch
        )
        val_m    = validate(model, val_loader, device, args.amp)
        scheduler.step()

        elapsed = time.time() - t0
        lr_head = optimiser.param_groups[1]["lr"]

        print(
            f"{epoch+1:>6}  {train_loss:>11.4f}  {val_m['rmse']:>9.4f}  "
            f"{val_m['abs_rel']:>8.4f}  {val_m['delta1']:>7.4f}  {lr_head:>9.2e}  {elapsed:.0f}s"
        )

        state = {
            "epoch": epoch + 1, "model": model.state_dict(),
            "optimiser": optimiser.state_dict(), "scheduler": scheduler.state_dict(),
            "best_rmse": best_rmse,
        }
        torch.save(state, save_dir / f"epoch_{epoch+1:03d}.pth")

        if val_m["rmse"] < best_rmse:
            best_rmse = val_m["rmse"]
            torch.save(state, save_dir / "best.pth")
            print(f"  ✓ New best RMSE: {best_rmse:.4f}")

    print(f"\nDone. Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
