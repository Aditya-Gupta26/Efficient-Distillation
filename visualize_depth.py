"""
Depth prediction visualization using the fine-tuned DPT head.

Loads the trained StudentWithDPT model (distilled Swin-Tiny backbone +
fine-tuned DPT depth head), runs inference on random NYU Depth V2 images,
and saves a side-by-side figure:
    Column 1 — RGB input
    Column 2 — Ground-truth depth (metres)
    Column 3 — Predicted depth (metres)

Usage:
    python visualize_depth.py \
        --student_checkpoint checkpoints/best.pth \
        --depth_checkpoint   checkpoints/depth_head/best.pth
    python visualize_depth.py --n 5 --seed 7
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.nyu_depth_dataset import NyuDepthDataset
from models.depth_model import StudentWithDPT

# ImageNet normalisation stats (must match the dataset loader)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalise(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor → (H,W,3) uint8 numpy array for display."""
    img = (t.cpu() * _STD + _MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_checkpoint", default="checkpoints/best.pth",
                   help="Distilled student backbone checkpoint (default: checkpoints/best.pth)")
    p.add_argument("--depth_checkpoint",   default="checkpoints/depth_head/best.pth",
                   help="Fine-tuned DPT depth head checkpoint (default: checkpoints/depth_head/best.pth)")
    p.add_argument("--nyu_root",  default="data/nyu_depth_v2")
    p.add_argument("--n",         type=int, default=10, help="Number of images (default: 10)")
    p.add_argument("--seed",      type=int, default=10)
    p.add_argument("--img_size",  type=int, default=224)
    p.add_argument("--output",    default="depth_predictions.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    # student_checkpoint → distilled Swin-Tiny backbone (weights from distillation run)
    # depth_checkpoint   → fine-tuned DPT head (weights from frozenBase_customHead.py)
    print(f"Student checkpoint : {args.student_checkpoint}")
    print(f"Depth checkpoint   : {args.depth_checkpoint}")

    # Build the model with the distilled student backbone
    model = StudentWithDPT(
        student_checkpoint=args.student_checkpoint,
        dpt_pretrained=False,
        freeze_student=True,
    )

    # Load the fine-tuned DPT depth head weights on top
    ckpt  = torch.load(args.depth_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ready — {total:,} params total, {trainable:,} trainable (DPT head)\n")

    # ── Previous timm fallback (kept for reference) ───────────────────────────
    # Used when we didn't have a valid distilled checkpoint:
    # model = StudentWithDPT(student_checkpoint=None, dpt_pretrained=False, freeze_student=True)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Pick random images ────────────────────────────────────────────────────
    dataset = NyuDepthDataset(root=args.nyu_root, split="val", img_size=args.img_size)
    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), k=min(args.n, len(dataset)))
    print(f"Visualising {len(indices)} random val images (seed={args.seed}) ...")

    # ── Inference ─────────────────────────────────────────────────────────────
    images, gt_depths, pred_depths = [], [], []

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            image  = sample["images"].unsqueeze(0).to(device)   # (1,3,H,W)
            depth  = sample["depths"]                           # (H,W) ground truth

            pred = model(image).squeeze().cpu()                 # (H,W) prediction

            images.append(sample["images"])
            gt_depths.append(depth)
            pred_depths.append(pred)

    # ── Plot ──────────────────────────────────────────────────────────────────
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]   # keep 2-D indexing for n=1

    fig.suptitle(
        "Fine-tuned DPT Depth Head — NYU Depth V2 Predictions\n"
        f"(Distilled Swin-Tiny backbone: {Path(args.student_checkpoint).name}  "
        f"+ DPT head: {Path(args.depth_checkpoint).name})",
        fontsize=13, fontweight="bold", y=1.002,
    )
    for ax, title in zip(axes[0], ["RGB Input", "Ground-truth Depth (m)", "Predicted Depth (m)"]):
        ax.set_title(title, fontsize=11, fontweight="bold")

    for row, (img_t, gt, pred) in enumerate(zip(images, gt_depths, pred_depths)):
        gt_np   = gt.numpy()
        pred_np = pred.numpy()

        valid = gt_np[gt_np > 0]
        vmin  = float(valid.min()) if len(valid) else 0.0
        vmax  = float(gt_np.max())

        # RGB
        axes[row, 0].imshow(denormalise(img_t))
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"idx {indices[row]}", fontsize=8,
                                rotation=0, labelpad=50, va="center")

        # Ground truth
        im_gt = axes[row, 1].imshow(gt_np, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 1].axis("off")
        plt.colorbar(im_gt, ax=axes[row, 1], fraction=0.046, pad=0.04, label="m")

        # Prediction — clip to GT range so colours are directly comparable
        im_pd = axes[row, 2].imshow(pred_np, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 2].axis("off")
        plt.colorbar(im_pd, ax=axes[row, 2], fraction=0.046, pad=0.04, label="m")

        # Print per-image stats under the prediction column
        mask = gt_np > 0
        rmse = float(np.sqrt(((pred_np[mask] - gt_np[mask]) ** 2).mean()))
        axes[row, 2].set_xlabel(
            f"pred mean={pred_np.mean():.2f}m   gt mean={gt_np[mask].mean():.2f}m   RMSE={rmse:.3f}m",
            fontsize=8,
        )

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {args.output}")


if __name__ == "__main__":
    main()
