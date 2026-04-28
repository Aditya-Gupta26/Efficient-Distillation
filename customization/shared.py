"""
shared.py — Utilities shared across all customization experiments.

Contains:
  - Depth metrics (RMSE, AbsRel, log-RMSE, SILog, δ1, δ2, δ3)
  - SILog training loss
  - StudentWithDPTUnfrozen: subclass with gradients flowing through the student
  - Checkpoint save / load
  - W&B depth preview logger (called every 50 epochs)
  - Shared training + validation loops
  - Fixed random seed for consistent 10-image evaluation across all runs

Nothing in this file modifies any existing project file.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project root on sys.path so we can import from models/, data/, utils/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.depth_model import StudentWithDPT
from data.nyu_depth_dataset import NyuDepthDataset, build_nyu_depth_dataloaders
from utils.device import (
    get_device,
    maybe_autocast,
    make_scaler,
    pin_memory_for,
    get_distributed_info,
)
from utils.distributed import (
    is_main_process,
    get_world_size,
    barrier,
    unwrap_model,
)
from torch.utils.data.distributed import DistributedSampler

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FIXED_SEED      = 42      # Seed for picking the same 10 val images in every experiment
N_EVAL_IMAGES   = 10      # Number of images used for visual comparison
SILOG_LAMBDA    = 0.85    # λ in SILog loss — standard value (AdaBins / DPT convention)

# ImageNet normalisation constants (must match NyuDepthDataset)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed validation indices — same 10 images used by every experiment
# ─────────────────────────────────────────────────────────────────────────────

def get_fixed_val_indices(dataset_len: int) -> list[int]:
    """Return the same N_EVAL_IMAGES val indices regardless of when/where it is called."""
    rng = random.Random(FIXED_SEED)
    return rng.sample(range(dataset_len), k=min(N_EVAL_IMAGES, dataset_len))


# ─────────────────────────────────────────────────────────────────────────────
# Depth metrics — industry-standard evaluation suite
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Compute 7 standard monocular depth estimation metrics.

    Args:
        pred   : predicted depth (any shape), values in metres, > 0
        target : ground-truth depth (same shape), 0 where invalid

    Returns dict with:
        rmse     — Root Mean Squared Error (metres)
        abs_rel  — Mean Absolute Relative Error
        log_rmse — Root Mean Squared Error in log space
        silog    — Scale-Invariant Logarithmic Error
        delta1   — % pixels with max(pred/gt, gt/pred) < 1.25
        delta2   — same threshold 1.25²
        delta3   — same threshold 1.25³
    """
    mask = (target > 0) & (pred > 1e-6)
    if mask.sum() == 0:
        return {k: 0.0 for k in ["rmse", "abs_rel", "log_rmse", "silog", "delta1", "delta2", "delta3"]}

    p, t = pred[mask], target[mask]

    rmse     = torch.sqrt(((p - t) ** 2).mean()).item()
    abs_rel  = (torch.abs(p - t) / t).mean().item()
    log_rmse = torch.sqrt(((torch.log(p) - torch.log(t)) ** 2).mean()).item()

    # SILog: scale-invariant log error (lower is better)
    d     = torch.log(p) - torch.log(t)
    silog = torch.sqrt(d.var() + 0.15 * d.mean() ** 2).item()

    # Threshold accuracy (higher is better)
    ratio  = torch.max(p / t, t / p)
    delta1 = (ratio < 1.25     ).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return dict(rmse=rmse, abs_rel=abs_rel, log_rmse=log_rmse,
                silog=silog, delta1=delta1, delta2=delta2, delta3=delta3)


# ─────────────────────────────────────────────────────────────────────────────
# SILog training loss
# ─────────────────────────────────────────────────────────────────────────────

def silog_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-Invariant Log loss — standard loss for monocular depth."""
    mask = (target > 0) & (pred > 1e-6)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    d = torch.log(pred[mask]) - torch.log(target[mask])
    return torch.sqrt(d.var() + (1.0 - SILOG_LAMBDA) * d.mean() ** 2 + 1e-8)


def reduce_metrics_dict(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    """
    Average scalar metrics across all distributed ranks.
    In single-process mode, returns metrics unchanged.
    """
    info = get_distributed_info()
    if not info["distributed"]:
        return metrics

    reduced = {}
    world_size = get_world_size()
    for k, v in metrics.items():
        t = torch.tensor(float(v), device=device, dtype=torch.float64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        t /= world_size
        reduced[k] = t.item()
    return reduced

def reduce_scalar_mean(value: float, device: torch.device) -> float:
    """
    Average one scalar across all distributed ranks.
    In single-process mode, returns the value unchanged.
    """
    info = get_distributed_info()
    if not info["distributed"]:
        return float(value)

    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    t /= get_world_size()
    return t.item()


# ─────────────────────────────────────────────────────────────────────────────
# StudentWithDPTUnfrozen — gradients flow through the student backbone
# ─────────────────────────────────────────────────────────────────────────────

class StudentWithDPTUnfrozen(StudentWithDPT):
    """
    Variant of StudentWithDPT where the student backbone is TRAINABLE.

    The parent class wraps the student's forward pass in torch.no_grad() and
    detaches the feature tensors — this is correct when the student is frozen
    (no gradients needed) but severs the gradient graph when we want to train
    the student.

    This subclass overrides forward() to remove no_grad / detach so that:
      - Gradients flow all the way back through the student.
      - The DPT head loss can update both the head AND the student backbone.

    The original checkpoint at checkpoints/best.pth is NEVER modified —
    only the in-memory copy of the weights is updated during training.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run the student backbone WITHOUT torch.no_grad() so that PyTorch
        # records operations for backpropagation through the student.
        features, _ = self.student(x)

        # Do NOT detach features — the gradient must flow from the DPT head
        # loss back through the feature tensors into the student weights.
        depth = self.dpt_head(features)

        # Upsample depth map to match input resolution (same as parent class)
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:],
                                  mode="bicubic", align_corners=False)
        return depth


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, save_dir: str | Path, is_best: bool) -> None:
    """Save latest.pth every epoch; overwrite best.pth only on improvement."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_dir / "latest.pth")
    if is_best:
        torch.save(state, save_dir / "best.pth")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> int:
    """
    Load a training checkpoint and restore model/optimiser/scheduler state.
    Returns the epoch to resume from (0 if no checkpoint exists).
    """
    path = Path(path)
    if not path.exists():
        return 0   # no checkpoint yet — start from epoch 0

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model"])
    optimiser.load_state_dict(ckpt["optimiser"])
    scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt.get("epoch", 0)
    print(f"[checkpoint] Resumed from {path}  (epoch {epoch}, best RMSE {ckpt.get('best_rmse', '?'):.4f})")
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
# W&B depth preview — logged every 50 epochs so we can track visual progress
# ─────────────────────────────────────────────────────────────────────────────

def log_depth_preview(
    model:   nn.Module,
    sample:  dict,           # one pre-loaded NyuDepthDataset sample
    device:  torch.device,
    epoch:   int,
    run,                     # wandb run object (or None)
    tag:     str = "depth_preview",
) -> None:
    """
    Run inference on a single fixed image and log a 3-panel figure to W&B.
    Called every 50 epochs so you can watch depth quality improve online.

    Panels: RGB input | Ground-truth depth | Predicted depth
    """
    if run is None:
        return

    import wandb

    model.eval()
    with torch.no_grad():
        image = sample["images"].unsqueeze(0).to(device)   # (1, 3, H, W)
        pred  = model(image).squeeze().cpu().numpy()       # (H, W)

    gt_np  = sample["depths"].numpy()                      # (H, W)
    rgb_np = ((sample["images"].cpu() * _STD + _MEAN)
              .clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    valid = gt_np[gt_np > 0]
    vmin, vmax = (float(valid.min()), float(gt_np.max())) if len(valid) else (0.0, 10.0)

    # Rotate 90° counter-clockwise so NYU images appear upright in W&B
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.rot90(rgb_np, k=1));                                          axes[0].set_title("RGB");                   axes[0].axis("off")
    axes[1].imshow(np.rot90(gt_np,  k=1), cmap="magma", vmin=vmin, vmax=vmax);     axes[1].set_title("GT depth");              axes[1].axis("off")
    axes[2].imshow(np.rot90(pred,   k=1), cmap="magma", vmin=vmin, vmax=vmax);     axes[2].set_title(f"Pred epoch {epoch}");   axes[2].axis("off")
    plt.tight_layout()

    run.log({tag: wandb.Image(fig)}, step=epoch)
    plt.close(fig)
    model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Shared training and validation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, scaler, device, grad_clip, amp, epoch):
    """One full pass over the training set. Returns mean SILog loss."""
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
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item()

        if is_main_process() and ((i + 1) % 20 == 0 or (i + 1) == n_batches):
            print(f"  epoch {epoch+1}  [{i+1:>3}/{n_batches}]  "
                  f"loss={loss.item():.4f}  ({time.time()-t0:.0f}s)",
                  end="\r", flush=True)

    if is_main_process():
        print()
    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, amp):
    """Full validation pass. Returns averaged depth metrics dict."""
    model.eval()
    sums = {k: 0.0 for k in ["rmse", "abs_rel", "log_rmse", "silog", "delta1", "delta2", "delta3"]}
    n = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        depths = batch["depths"].to(device, non_blocking=True)
        with maybe_autocast(device, amp):
            pred = model(images).squeeze(1)
        m = compute_depth_metrics(pred, depths)
        b = images.size(0)
        for k in sums:
            sums[k] += m[k] * b
        n += b

    return {k: v / n for k, v in sums.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Master training orchestrator — called by each experiment script
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    model:            nn.Module,
    train_loader,
    val_loader,
    val_dataset:      NyuDepthDataset,
    optimiser:        torch.optim.Optimizer,
    scheduler,
    device:           torch.device,
    args,
    run_name:         str,
    wandb_run=None,
) -> None:
    """
    Shared training loop used by all 4 experiment variants.

    Handles:
      - Resuming from latest.pth if it exists in args.save_dir
      - Epoch progress printing
      - best.pth / latest.pth checkpoint saving
      - W&B metric logging every epoch
      - W&B depth preview image every 50 epochs
    """
    save_dir  = Path(args.save_dir)
    scaler    = make_scaler(device, args.amp)
    best_rmse = float("inf")

    # Fixed preview sample: the same image for all epochs and all runs
    preview_idx    = get_fixed_val_indices(len(val_dataset))[0]
    preview_sample = val_dataset[preview_idx]

    # Resume if a checkpoint already exists (handles interrupted runs)
    start_epoch = load_checkpoint(
        save_dir / "latest.pth", model, optimiser, scheduler
    )

    # Reload best_rmse from best.pth so we don't overwrite a better model
    best_ckpt = save_dir / "best.pth"
    if best_ckpt.exists():
        best_rmse = torch.load(best_ckpt, map_location="cpu",
                               weights_only=False).get("best_rmse", float("inf"))

    if is_main_process():
        print(f"\n{'Epoch':>6}  {'Loss':>8}  {'RMSE':>7}  {'AbsRel':>7}  "
              f"{'SILog':>7}  {'δ1':>6}  {'LR':>9}  Time")
        print("─" * 72)

    for epoch in range(start_epoch, args.epochs):
        train_sampler = getattr(train_loader, "sampler", None)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimiser, scaler,
            device, args.grad_clip, args.amp, epoch,
        )
        train_loss = reduce_scalar_mean(train_loss, device)
        val_m = validate(model, val_loader, device, args.amp)
        val_m = reduce_metrics_dict(val_m, device)
        scheduler.step()
        elapsed  = time.time() - t0
        lr_now   = scheduler.get_last_lr()[0]
        is_best  = val_m["rmse"] < best_rmse

        if is_main_process():
            print(f"{epoch+1:>6}  {train_loss:>8.4f}  {val_m['rmse']:>7.4f}  "
                  f"{val_m['abs_rel']:>7.4f}  {val_m['silog']:>7.4f}  "
                  f"{val_m['delta1']:>6.4f}  {lr_now:>9.2e}  {elapsed:.0f}s",
                  flush=True)

        if is_best:
            best_rmse = val_m["rmse"]
            if is_main_process():
                print(f"  ✓ New best RMSE {best_rmse:.4f} → {save_dir}/best.pth", flush=True)

        # Save checkpoint (always latest.pth, best.pth only on improvement)
        if is_main_process():
            state = dict(
                epoch=epoch + 1,
                model=unwrap_model(model).state_dict(),
                optimiser=optimiser.state_dict(),
                scheduler=scheduler.state_dict(),
                best_rmse=best_rmse,
            )
            save_checkpoint(state, save_dir, is_best)

        # W&B: log metrics every epoch
        if wandb_run is not None and is_main_process():
            wandb_run.log({
                "train/loss":      train_loss,
                "val/rmse":        val_m["rmse"],
                "val/abs_rel":     val_m["abs_rel"],
                "val/log_rmse":    val_m["log_rmse"],
                "val/silog":       val_m["silog"],
                "val/delta1":      val_m["delta1"],
                "val/delta2":      val_m["delta2"],
                "val/delta3":      val_m["delta3"],
                "train/lr":        lr_now,
            }, step=epoch + 1)

        # W&B: log depth preview image every 10 epochs (tracks visual progress)
        if wandb_run is not None and is_main_process() and ((epoch + 1) % 10 == 0 or epoch == start_epoch):
            log_depth_preview(model, preview_sample, device, epoch + 1,
                              wandb_run, tag="depth_preview")

    if is_main_process():
        print(f"\nDone — {run_name}  |  Best RMSE: {best_rmse:.4f}")
    if wandb_run is not None and is_main_process():
        wandb_run.summary["best_rmse"] = best_rmse
        wandb_run.finish()

    barrier()
