"""
Validation: distilled student + DPT depth head on NYU-Depth V2.

This is the primary evaluation script.  It demonstrates that:
  1. The distilled Swin-Tiny student backbone produces correct feature maps.
  2. The DPT custom depth head attached to it generates accurate depth maps.
  3. The full StudentWithDPT pipeline performs correctly on NYU-Depth V2.

Metrics reported (standard NYU-Depth V2 benchmark):
    RMSE      root mean squared error (metres)
    AbsRel    mean absolute relative error
    SqRel     mean squared relative error
    log10     mean |log10(pred) - log10(gt)|
    SILog     scale-invariant log error
    δ1        % pixels where max(pred/gt, gt/pred) < 1.25
    δ2        % pixels where max(pred/gt, gt/pred) < 1.25²
    δ3        % pixels where max(pred/gt, gt/pred) < 1.25³

Outputs:
    - Metrics table printed to stdout
    - PNG figure with 10 sample predictions saved to --output_dir
      (each row: RGB image | Ground-truth depth | Predicted depth)

Usage:
    python finetuning/base_validation.py \
        --student_checkpoint checkpoints/epoch_069.pth \
        --depth_checkpoint   checkpoints/depth/best.pth \
        --nyu_root           /path/to/nyu_depth_v2 \
        --output_dir         finetuning/outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.depth_model import StudentWithDPT
from data.nyu_depth_dataset import build_nyu_depth_dataloaders, NyuDepthDataset
from utils.device import get_device, maybe_autocast, pin_memory_for

# matplotlib imported lazily below to avoid hard dep at module level


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

SILOG_LAMBDA = 0.85


def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    All standard NYU-Depth V2 benchmark metrics.

    Args:
        pred  : (B, H, W) predicted depth, metres, clamped > 0.
        target: (B, H, W) ground-truth depth, metres.

    Returns:
        dict of metric_name → float  (batch-level, not pixel-level average).
    """
    mask = (target > 0) & (pred > 1e-6)
    if mask.sum() == 0:
        return {k: 0.0 for k in ["rmse", "abs_rel", "sq_rel", "log10", "silog", "delta1", "delta2", "delta3"]}

    p, t = pred[mask], target[mask]

    # RMSE
    rmse = torch.sqrt(((p - t) ** 2).mean()).item()

    # AbsRel
    abs_rel = (torch.abs(p - t) / t).mean().item()

    # SqRel
    sq_rel = (((p - t) ** 2) / t).mean().item()

    # log10
    log10 = torch.abs(torch.log10(p) - torch.log10(t)).mean().item()

    # SILog
    d = torch.log(p) - torch.log(t)
    silog = torch.sqrt(d.var() + (1.0 - SILOG_LAMBDA) * d.mean() ** 2 + 1e-8).item()

    # Threshold accuracies δ1, δ2, δ3
    ratio  = torch.max(p / t, t / p)
    delta1 = (ratio < 1.25     ).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "rmse":    rmse,
        "abs_rel": abs_rel,
        "sq_rel":  sq_rel,
        "log10":   log10,
        "silog":   silog,
        "delta1":  delta1,
        "delta2":  delta2,
        "delta3":  delta3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(student_checkpoint: str, depth_checkpoint: str) -> StudentWithDPT:
    model = StudentWithDPT(
        student_checkpoint=student_checkpoint,
        dpt_pretrained=False,   # weights come from depth_checkpoint
        freeze_student=True,
    )

    ckpt = torch.load(depth_checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] Missing keys  : {len(missing)}")
    if unexpected:
        print(f"[load] Unexpected keys: {len(unexpected)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"StudentWithDPT loaded — {total:,} parameters")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Full validation pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(model, val_loader, device, amp) -> dict:
    model.eval()

    accum = {k: 0.0 for k in ["rmse", "abs_rel", "sq_rel", "log10", "silog", "delta1", "delta2", "delta3"]}
    n = 0

    for batch in val_loader:
        images = batch["images"].to(device, non_blocking=True)
        depths = batch["depths"].to(device, non_blocking=True)

        with maybe_autocast(device, amp):
            pred = model(images).squeeze(1)   # (B, H, W)

        m = compute_depth_metrics(pred, depths)
        b = images.size(0)
        for k in accum:
            accum[k] += m[k] * b
        n += b

    return {k: v / n for k, v in accum.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation: 10 predictions
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm_rgb(tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor → (H, W, 3) uint8 numpy array."""
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)   # HWC
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def save_prediction_grid(
    model:     StudentWithDPT,
    dataset:   NyuDepthDataset,
    device:    torch.device,
    out_path:  Path,
    n:         int = 10,
    amp:       bool = True,
):
    """
    Sample n images from the dataset, run the model, and save a figure with
    3 columns per row: RGB  |  Ground-truth depth  |  Predicted depth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    model.eval()

    # Evenly spaced indices across the val set so we get diverse scenes
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(
        "StudentWithDPT — NYU-Depth V2 Validation\n"
        "(distilled Swin-Tiny backbone  +  DPT depth head)",
        fontsize=14, fontweight="bold", y=1.005,
    )

    col_titles = ["RGB Input", "Ground-truth Depth", "Predicted Depth"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        img_t   = sample["images"].unsqueeze(0).to(device)   # (1,3,H,W)
        depth_t = sample["depths"]                            # (H,W)

        with maybe_autocast(device, amp):
            pred_t = model(img_t).squeeze()                   # (H,W)

        rgb  = denorm_rgb(sample["images"])
        gt   = depth_t.cpu().numpy()
        pred = pred_t.cpu().float().numpy()

        # Shared depth range for fair comparison
        vmin = float(gt[gt > 0].min()) if (gt > 0).any() else 0.0
        vmax = float(gt.max())

        # Column 0: RGB
        axes[row, 0].imshow(rgb)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=9, rotation=0, labelpad=60, va="center")

        # Column 1: GT depth
        im_gt = axes[row, 1].imshow(gt, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 1].axis("off")
        plt.colorbar(im_gt, ax=axes[row, 1], fraction=0.046, pad=0.04, label="depth (m)")

        # Column 2: Predicted depth
        im_pr = axes[row, 2].imshow(pred, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 2].axis("off")
        plt.colorbar(im_pr, ax=axes[row, 2], fraction=0.046, pad=0.04, label="depth (m)")

        # Per-sample metrics as subtitle on prediction column
        mask = (depth_t > 0) & (pred_t.cpu() > 1e-6)
        if mask.sum() > 0:
            p_m, t_m = pred_t.cpu()[mask], depth_t[mask]
            rmse_s   = torch.sqrt(((p_m - t_m) ** 2).mean()).item()
            rel_s    = (torch.abs(p_m - t_m) / t_m).mean().item()
            axes[row, 2].set_xlabel(
                f"RMSE={rmse_s:.3f} m  |  AbsRel={rel_s:.3f}", fontsize=8
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction grid saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics: dict, checkpoint: str) -> None:
    w = 62
    print()
    print("═" * w)
    print("  NYU-Depth V2 — Evaluation Results")
    print(f"  Checkpoint: {Path(checkpoint).name}")
    print("═" * w)

    # Lower is better
    print("  ── Error Metrics (lower is better) ──────────────────")
    print(f"  {'RMSE':<20} {metrics['rmse']:>10.4f}  m")
    print(f"  {'AbsRel':<20} {metrics['abs_rel']:>10.4f}")
    print(f"  {'SqRel':<20} {metrics['sq_rel']:>10.4f}")
    print(f"  {'log10':<20} {metrics['log10']:>10.4f}")
    print(f"  {'SILog':<20} {metrics['silog']:>10.4f}")
    print()

    # Higher is better
    print("  ── Accuracy Metrics (higher is better) ──────────────")
    print(f"  {'δ1  (thr=1.25)':<20} {metrics['delta1']:>10.4f}  ({metrics['delta1']*100:.2f}%)")
    print(f"  {'δ2  (thr=1.25²)':<20} {metrics['delta2']:>10.4f}  ({metrics['delta2']*100:.2f}%)")
    print(f"  {'δ3  (thr=1.25³)':<20} {metrics['delta3']:>10.4f}  ({metrics['delta3']*100:.2f}%)")
    print("═" * w)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate StudentWithDPT on NYU-Depth V2 and save 10 prediction images"
    )
    p.add_argument("--student_checkpoint", required=True,
                   help="Distillation checkpoint (e.g. checkpoints/epoch_069.pth)")
    p.add_argument("--depth_checkpoint", required=True,
                   help="Depth training checkpoint (e.g. checkpoints/depth/best.pth)")
    p.add_argument("--nyu_root",    required=True, help="NYU-Depth V2 root directory")
    p.add_argument("--output_dir",  default="finetuning/outputs",
                   help="Directory to save the prediction grid PNG")
    p.add_argument("--batch_size",  type=int,  default=16)
    p.add_argument("--num_workers", type=int,  default=8)
    p.add_argument("--img_size",    type=int,  default=224)
    p.add_argument("--n_images",    type=int,  default=10,
                   help="Number of prediction images to visualise")
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--no_amp",      action="store_false", dest="amp")
    p.add_argument("--split",       default="val", choices=["train", "val"],
                   help="Dataset split to evaluate on (default: val)")
    return p.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Device : {device}")
    print(f"Split  : {args.split}")

    # ── Load model ──────────────────────────────────────────────────────────
    model = load_model(args.student_checkpoint, args.depth_checkpoint).to(device)

    # ── Verify parameter status ─────────────────────────────────────────────
    student_trainable = sum(p.numel() for p in model.student.parameters() if p.requires_grad)
    head_trainable    = sum(p.numel() for p in model.dpt_head.parameters() if p.requires_grad)
    student_total     = sum(p.numel() for p in model.student.parameters())
    head_total        = sum(p.numel() for p in model.dpt_head.parameters())

    print()
    print("  Parameter status:")
    print(f"    Student backbone : {student_total:>10,} params  |  trainable: {student_trainable:,}")
    print(f"    DPT head         : {head_total:>10,} params  |  trainable: {head_trainable:,}")
    print()

    # ── Data loaders ────────────────────────────────────────────────────────
    _, val_loader = build_nyu_depth_dataloaders(
        root=args.nyu_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Val set: {len(val_loader.dataset):,} samples")

    # ── Full validation ──────────────────────────────────────────────────────
    print("Running validation …")
    metrics = run_validation(model, val_loader, device, args.amp)
    print_metrics_table(metrics, args.depth_checkpoint)

    # ── Prediction visualisation ─────────────────────────────────────────────
    val_dataset = NyuDepthDataset(args.nyu_root, split=args.split, img_size=args.img_size)
    out_path    = Path(args.output_dir) / "predictions_10.png"

    print(f"Generating {args.n_images} prediction images …")
    save_prediction_grid(
        model=model,
        dataset=val_dataset,
        device=device,
        out_path=out_path,
        n=args.n_images,
        amp=args.amp,
    )

    print("Validation complete.")


if __name__ == "__main__":
    main()
