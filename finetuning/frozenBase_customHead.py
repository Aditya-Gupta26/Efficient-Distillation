"""
Frozen-backbone fine-tuning: student weights locked, DPT head trains.

Trains the DPT depth head on top of the frozen distilled student backbone
on NYU-Depth V2.  Student weights never change; only the DPT head learns.

Usage:
    # First run — train DPT head from scratch:
    python finetuning/frozenBase_customHead.py \
        --student_checkpoint checkpoints/epoch_069.pth \
        --nyu_root           data/nyu_depth_v2 \
        --epochs             30 \
        --save_dir           checkpoints/depth_head

    # Resume from a saved checkpoint:
    python finetuning/frozenBase_customHead.py \
        --student_checkpoint checkpoints/epoch_069.pth \
        --depth_checkpoint   checkpoints/depth_head/best.pth \
        --nyu_root           data/nyu_depth_v2 \
        --save_dir           checkpoints/depth_head
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
# Architecture verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_architecture(checkpoint_path: str) -> bool:
    """
    Prove that the distilled student checkpoint uses the exact same Swin-Tiny
    architecture as the timm pretrained reference model.

    This is important because:
      - If the architectures differ, the frozen-backbone fine-tuning approach
        would be invalid (you would be using the wrong model).
      - If they match, then any problem is purely in the WEIGHTS (e.g. NaN from
        a diverged distillation run) — not in the model design.

    Compares every parameter name and tensor shape between:
      A) The distilled checkpoint's state dict  (architecture only — no weights applied)
      B) A freshly built timm Swin-Tiny model   (the authoritative reference)
    """
    from models.student import SwinStudentTiny

    print("\n" + "=" * 65)
    print("  ARCHITECTURE VERIFICATION")
    print("  Distilled checkpoint  vs  timm pretrained Swin-Tiny")
    print("=" * 65)

    # ── Load checkpoint keys + shapes (do NOT apply weights to any model) ─────
    ckpt       = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt.get("student", ckpt)
    # Strip torch.compile prefix so key names are comparable
    ckpt_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt_state.items()}

    # ── Reference: timm Swin-Tiny with num_classes=1000 ──────────────────────
    # This is the exact same model class used throughout this project.
    ref_state = SwinStudentTiny(pretrained=False, num_classes=1000).state_dict()

    # ── Compare ───────────────────────────────────────────────────────────────
    ckpt_keys = set(ckpt_state.keys())
    ref_keys  = set(ref_state.keys())

    missing_in_ckpt = ref_keys  - ckpt_keys   # in timm but absent from checkpoint
    extra_in_ckpt   = ckpt_keys - ref_keys     # in checkpoint but unknown to timm

    shape_ok, shape_bad = [], []
    for k in ckpt_keys & ref_keys:
        if ckpt_state[k].shape == ref_state[k].shape:
            shape_ok.append(k)
        else:
            shape_bad.append((k, ckpt_state[k].shape, ref_state[k].shape))

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"  timm reference parameters  : {len(ref_keys)}")
    print(f"  Checkpoint parameters      : {len(ckpt_keys)}")
    print(f"  Matching names             : {len(ckpt_keys & ref_keys)}")
    print(f"  Shape matches              : {len(shape_ok)}")
    print(f"  Shape mismatches           : {len(shape_bad)}")
    print(f"  Keys missing from ckpt     : {len(missing_in_ckpt)}")
    print(f"  Unexpected keys in ckpt    : {len(extra_in_ckpt)}")

    if missing_in_ckpt:
        print(f"\n  Missing keys (sample) : {sorted(missing_in_ckpt)[:4]}")
    if extra_in_ckpt:
        print(f"  Extra   keys (sample) : {sorted(extra_in_ckpt)[:4]}")
    if shape_bad:
        print(f"\n  Shape mismatches (sample):")
        for k, cs, rs in shape_bad[:4]:
            print(f"    {k}:  checkpoint {cs}  ≠  timm {rs}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    is_match = (len(shape_bad) == 0 and len(missing_in_ckpt) == 0 and len(extra_in_ckpt) == 0)

    if is_match:
        print(f"\n  ✓  ARCHITECTURE MATCH CONFIRMED")
        print(f"     All {len(shape_ok)} parameters have identical names and shapes.")
        print(f"     The distilled model IS a standard Swin-Tiny — architecture is correct.")
        print(f"     The only problem is the WEIGHTS (NaN from a diverged training run).")
        print(f"     → Get best.pth from the distillation run to fix the weights.")
    else:
        print(f"\n  ✗  ARCHITECTURE MISMATCH — see mismatches above.")

    print("=" * 65 + "\n")
    return is_match


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(student_checkpoint: str | None, depth_checkpoint: str | None) -> StudentWithDPT:
    # student_checkpoint=None  → timm pretrained Swin-Tiny (pipeline testing)
    # student_checkpoint=path  → distilled student from that checkpoint
    #                            (will raise RuntimeError if checkpoint has NaN weights)

    # Always verify architecture first when a checkpoint is provided.
    # This proves the distilled model has the correct Swin-Tiny structure
    # before we attempt to load its weights (which may be NaN / corrupted).
    if student_checkpoint is not None:
        verify_architecture(student_checkpoint)

    model = StudentWithDPT(
        student_checkpoint=student_checkpoint,
        dpt_pretrained=False,
        freeze_student=True,
    )

    if depth_checkpoint is not None:
        ckpt  = torch.load(depth_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[load] Missing keys  : {len(missing)}  (e.g. {missing[:3]})")
        if unexpected:
            print(f"[load] Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        print(f"Resumed from depth checkpoint: {depth_checkpoint}")
    else:
        print("Starting fresh — DPT head randomly initialised (architecture: Intel/dpt-swinv2-tiny-256)")

    for p in model.student.parameters():
        p.requires_grad = False
    model.student.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Model ready — total: {total:,}  |  trainable (DPT head only): {trainable:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, scaler, device, grad_clip, amp, epoch):
    model.train()
    model.student.eval()

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
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item()

        # Print progress every 10 batches so the user can see it's running
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch+1}  [{i+1:>3}/{n_batches}]  "
                f"loss={loss.item():.4f}  ({elapsed:.0f}s)",
                end="\r", flush=True,
            )

    print()  # newline after \r progress
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
    p = argparse.ArgumentParser(description="Fine-tune DPT head (student frozen)")
    p.add_argument("--student_checkpoint", default="checkpoints/best.pth",
                   help="Distilled student checkpoint (.pth). Always required — no fallback to timm.")
    p.add_argument("--depth_checkpoint",   default=None,
                   help="Resume from a prior depth checkpoint. Omit to train from scratch.")
    p.add_argument("--nyu_root",      required=True)
    p.add_argument("--save_dir",      default="checkpoints/depth_head")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--lr_min",        type=float, default=1e-6)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--batch_size",    type=int,   default=8,
                   help="Default 8 — safe for MPS; increase to 16 on CUDA")
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--amp",           action="store_true", default=True)
    p.add_argument("--no_amp",        action="store_false", dest="amp")
    return p.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Device : {device}")

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
    print(f"Batch size: {args.batch_size}  |  batches/epoch: {len(train_loader)}")

    model     = load_model(args.student_checkpoint, args.depth_checkpoint).to(device)

    # ── Previous timm fallback (kept for reference) ───────────────────────────
    # Used during development when epoch_069.pth had NaN weights.
    # Now that best.pth is valid, we always use the distilled student directly.
    #
    # student_ckpt = None if (args.student_checkpoint is None
    #                         or args.student_checkpoint.lower() == "none") \
    #                else args.student_checkpoint
    # model = load_model(student_ckpt, args.depth_checkpoint).to(device)
    # ─────────────────────────────────────────────────────────────────────────
    optimiser = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr_min)
    scaler    = make_scaler(device, args.amp)
    best_rmse = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val RMSE':>9}  {'AbsRel':>8}  {'δ1':>7}  {'LR':>9}  Time")
    print("─" * 72)

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimiser, scaler, device,
            args.grad_clip, args.amp, epoch,
        )
        val_m   = validate(model, val_loader, device, args.amp)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"{epoch+1:>6}  {train_loss:>11.4f}  {val_m['rmse']:>9.4f}  "
            f"{val_m['abs_rel']:>8.4f}  {val_m['delta1']:>7.4f}  {lr_now:>9.2e}  {elapsed:.0f}s"
        )

        state = {
            "epoch": epoch + 1, "model": model.state_dict(),
            "optimiser": optimiser.state_dict(), "scheduler": scheduler.state_dict(),
            "best_rmse": best_rmse,
        }
        # Overwrite latest.pth each epoch to avoid filling disk with per-epoch files
        torch.save(state, save_dir / "latest.pth")

        if val_m["rmse"] < best_rmse:
            best_rmse = val_m["rmse"]
            torch.save(state, save_dir / "best.pth")
            print(f"  ✓ New best RMSE: {best_rmse:.4f}  →  {save_dir}/best.pth")

    print(f"\nDone. Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
