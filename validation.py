"""
Quick visual test: distilled Swin-Tiny student + DPT depth head on 10 NYU images.

Loads the distilled student from a checkpoint, attaches the pretrained DPT depth
head, runs inference on 10 samples from the NYU-Depth V2 val set, prints per-image
metrics, and saves a figure (RGB | Ground-truth depth | Predicted depth).

Usage:
    python validation.py \
        --student_checkpoint checkpoints/epoch_069.pth \
        --nyu_root           data/nyu_depth_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.depth_model import StudentWithDPT
from data.nyu_depth_dataset import NyuDepthDataset
from utils.device import get_device


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

N_IMAGES = 10


def denorm(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().numpy().transpose(1, 2, 0) * _STD + _MEAN
    return np.clip(img, 0, 1)


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor):
    mask = (gt > 0) & (pred > 1e-6)
    p, t = pred[mask], gt[mask]
    rmse    = torch.sqrt(((p - t) ** 2).mean()).item()
    abs_rel = (torch.abs(p - t) / t).mean().item()
    delta1  = (torch.max(p / t, t / p) < 1.25).float().mean().item()
    return rmse, abs_rel, delta1


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate distilled student + DPT head on 10 NYU-Depth V2 images"
    )
    p.add_argument("--student_checkpoint", required=True,
                   help="Distillation checkpoint, e.g. checkpoints/epoch_069.pth")
    p.add_argument("--depth_checkpoint", default=None,
                   help="Trained depth head checkpoint (e.g. checkpoints/depth_head/best.pth). "
                        "Omit to use raw pretrained DPT head (will produce black images until trained).")
    p.add_argument("--nyu_root", required=True,
                   help="NYU-Depth V2 root dir (must contain val/rgb and val/depth)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--output",   default="validation_output.png",
                   help="Path to save the output figure")
    p.add_argument("--split",    default="val", choices=["train", "val"])
    return p.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    model = StudentWithDPT(
        student_checkpoint=args.student_checkpoint,
        # dpt_pretrained=(args.depth_checkpoint is None),
        dpt_pretrained=True,
        freeze_student=True,
    )
    if args.depth_checkpoint is not None:
        ckpt  = torch.load(args.depth_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print(f"Loaded trained depth head: {args.depth_checkpoint}")
    else:
        print("WARNING: no --depth_checkpoint given — DPT head is untrained on this student.")
        print("         Run finetuning/frozenBase_customHead.py first.")
    model.to(device).eval()

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = NyuDepthDataset(args.nyu_root, split=args.split, img_size=args.img_size)
    indices = np.linspace(0, len(dataset) - 1, N_IMAGES, dtype=int)
    print(f"Dataset ({args.split}): {len(dataset)} samples")
    print(f"Evaluating sample indices: {indices.tolist()}\n")

    # ── Inference + per-image metrics ────────────────────────────────────────
    print(f"{'#':>3}  {'idx':>5}  {'RMSE (m)':>10}  {'AbsRel':>8}  {'δ1':>7}")
    print("─" * 42)

    rows = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[int(idx)]
            img_t   = sample["images"].unsqueeze(0).to(device)
            depth_t = sample["depths"]              # (H, W) ground truth

            pred_t = model(img_t).squeeze().cpu()   # (H, W) predicted
            pred_t = pred_t.clamp(min=1e-6)

            rmse, abs_rel, delta1 = compute_metrics(pred_t, depth_t)
            print(f"{i+1:>3}  {idx:>5}  {rmse:>10.4f}  {abs_rel:>8.4f}  {delta1:>7.4f}")
            rows.append((sample["images"], depth_t, pred_t, idx, rmse, abs_rel))

    print("─" * 42)

    # ── Save figure: 10 rows × 3 columns (RGB | GT | Predicted) ─────────────
    fig, axes = plt.subplots(N_IMAGES, 3, figsize=(13, 4 * N_IMAGES))
    fig.suptitle(
        "Distilled Swin-Tiny + DPT Depth Head  —  NYU-Depth V2",
        fontsize=15, fontweight="bold", y=1.002,
    )
    for ax, title in zip(axes[0], ["RGB Input", "Ground-truth Depth (m)", "Predicted Depth (m)"]):
        ax.set_title(title, fontsize=12, fontweight="bold")

    for row_idx, (img_t, gt, pred, idx, rmse, abs_rel) in enumerate(rows):
        rgb     = denorm(img_t)
        gt_np   = gt.numpy()
        pred_np = pred.numpy()

        vmin = float(gt_np[gt_np > 0].min()) if (gt_np > 0).any() else 0.0
        vmax = float(gt_np.max())

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_ylabel(f"sample {idx}", fontsize=9, labelpad=4)
        axes[row_idx, 0].axis("off")

        im = axes[row_idx, 1].imshow(gt_np, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row_idx, 1].axis("off")
        plt.colorbar(im, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        im = axes[row_idx, 2].imshow(pred_np, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row_idx, 2].axis("off")
        plt.colorbar(im, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
        axes[row_idx, 2].set_xlabel(
            f"RMSE={rmse:.3f} m  |  AbsRel={abs_rel:.3f}", fontsize=9
        )

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out}")


if __name__ == "__main__":
    main()
