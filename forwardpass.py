"""
Forward-pass demo: distilled Swin-Tiny student + pretrained Intel DPT head.

Loads the distilled student backbone from a checkpoint, attaches the
*original* pretrained DPT depth head (Intel/dpt-swinv2-tiny-256 weights,
no fine-tuning), picks 10 random images from the NYU Depth V2 val split,
runs inference, and saves a figure:
    Column 1 — Original RGB image
    Column 2 — Ground-truth depth map (metres)
    Column 3 — Predicted depth map

Note: the DPT head here uses pretrained Intel weights that were trained on
SwinV2-Tiny features, not our Swin-Tiny student.  Predictions will likely
look noisy / mis-scaled — this is expected and illustrates *why* fine-tuning
the depth head (frozenBase_customHead.py) is necessary.

Usage:
    python forwardpass.py
    python forwardpass.py --student_checkpoint checkpoints/epoch_069.pth
    python forwardpass.py --nyu_root data/nyu_depth_v2 --seed 7
"""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.nyu_depth_dataset import NyuDepthDataset
from models.student import SwinStudentTiny
from models.dpt_head_original import DPTDepthHead   # pretrained Intel head, unmodified
from utils.device import get_device

# ── ImageNet normalisation stats (must match NyuDepthDataset) ────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalise(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised float tensor → (H, W, 3) uint8 numpy array."""
    img = (t.cpu() * _STD + _MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_student(checkpoint_path: str) -> SwinStudentTiny:
    """Load distilled Swin-Tiny student weights from a distillation checkpoint."""
    student = SwinStudentTiny(pretrained=False, num_classes=0)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Distillation checkpoints written by utils/checkpoint.py store weights under "student"
    state = ckpt["student"] if "student" in ckpt else ckpt
    # Strip torch.compile prefix if checkpoint came from a compiled model
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing:
        non_head = [k for k in missing if "head" not in k]
        if non_head:
            print(f"  [WARN] {len(non_head)} unexpected missing keys: {non_head[:3]}")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected keys in checkpoint.")

    print(f"Distilled student loaded from: {checkpoint_path}")
    return student


def load_model(student_checkpoint: str, device: torch.device):
    """
    Build the full inference pipeline:
        distilled Swin-Tiny (frozen) + pretrained Intel DPT head.
    """
    print("Loading distilled Swin-Tiny student …")
    student = load_student(student_checkpoint).to(device).eval()

    # Freeze student — weights must not change
    for p in student.parameters():
        p.requires_grad = False

    print("Loading pretrained Intel DPT depth head (Intel/dpt-swinv2-tiny-256) …")
    dpt_head = DPTDepthHead(pretrained=True).to(device).eval()

    total_s = sum(p.numel() for p in student.parameters())
    total_d = sum(p.numel() for p in dpt_head.parameters())
    print(f"  Student params : {total_s:,}  (frozen)")
    print(f"  DPT head params: {total_d:,}  (pretrained Intel weights)")
    return student, dpt_head


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    student:  SwinStudentTiny,
    dpt_head: DPTDepthHead,
    images:   torch.Tensor,     # (B, 3, H, W) normalised, on CPU
    device:   torch.device,
) -> torch.Tensor:              # (B, 1, H, W) predicted depth, on CPU
    x = images.to(device)

    # Student backbone: returns 4 NCHW feature maps (1/4 … 1/32 resolution)
    features, _ = student(x)

    # DPT neck + head: produces a coarse depth map.
    # dpt_head_original uses DPTDepthEstimationHead.forward() which internally
    # squeezes the channel dim → output is (B, H, W).  Re-add it so the rest
    # of the pipeline always works with (B, 1, H, W).
    depth = dpt_head(features)
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)   # (B, H, W) → (B, 1, H, W)

    # Upsample to original input resolution
    if depth.shape[-2:] != x.shape[-2:]:
        depth = F.interpolate(depth, size=x.shape[-2:],
                              mode="bicubic", align_corners=False)
    return depth.cpu()


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_figure(
    images:    list[torch.Tensor],   # list of (3, H, W)
    gt_depths: list[torch.Tensor],   # list of (H, W)
    pd_depths: list[torch.Tensor],   # list of (1, H, W)
    indices:   list[int],
    out_path:  Path,
) -> None:
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(
        "Distilled Swin-Tiny + Pretrained Intel DPT Head\nNYU Depth V2 — Forward Pass",
        fontsize=14, fontweight="bold", y=1.002,
    )
    for ax, title in zip(axes[0], ["RGB Input", "Ground-truth Depth (m)", "Predicted Depth (m)"]):
        ax.set_title(title, fontsize=12, fontweight="bold")

    for row, (img_t, gt, pd) in enumerate(zip(images, gt_depths, pd_depths)):
        rgb  = denormalise(img_t)
        gt_np = gt.numpy()
        pd_np = pd[0].numpy()

        # Shared colour range from ground truth for a fair visual comparison
        valid = gt_np[gt_np > 0]
        vmin  = float(valid.min()) if len(valid) else 0.0
        vmax  = float(gt_np.max())

        # Column 0: RGB
        axes[row, 0].imshow(rgb)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"sample {indices[row]}", fontsize=9,
                                rotation=0, labelpad=60, va="center")

        # Column 1: ground-truth depth
        im_gt = axes[row, 1].imshow(gt_np, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 1].axis("off")
        plt.colorbar(im_gt, ax=axes[row, 1], fraction=0.046, pad=0.04, label="m")

        # Column 2: predicted depth (clipped to GT range for display)
        im_pd = axes[row, 2].imshow(pd_np, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 2].axis("off")
        plt.colorbar(im_pd, ax=axes[row, 2], fraction=0.046, pad=0.04, label="m")
        axes[row, 2].set_xlabel(
            f"pred mean={pd_np.mean():.2f}m  gt mean={gt_np[gt_np>0].mean():.2f}m",
            fontsize=8,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Forward-pass demo: distilled student + pretrained DPT head")
    p.add_argument("--student_checkpoint", default="checkpoints/epoch_069.pth",
                   help="Distilled student checkpoint (default: checkpoints/epoch_069.pth)")
    p.add_argument("--nyu_root",  default="data/nyu_depth_v2",
                   help="NYU Depth V2 root directory (default: data/nyu_depth_v2)")
    p.add_argument("--split",     default="val", choices=["train", "val"])
    p.add_argument("--n",         type=int, default=10,
                   help="Number of random images to visualise (default: 10)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--img_size",  type=int, default=224)
    p.add_argument("--output",    default="forwardpass_output.png")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    print(f"Device: {device}\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = NyuDepthDataset(root=args.nyu_root, split=args.split, img_size=args.img_size)
    print(f"Dataset ({args.split}): {len(dataset)} samples")

    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), k=min(args.n, len(dataset)))
    print(f"Selected indices : {indices}\n")

    images, gt_depths = [], []
    for idx in indices:
        sample = dataset[idx]
        images.append(sample["images"])    # (3, H, W)
        gt_depths.append(sample["depths"]) # (H, W)

    # ── Model ────────────────────────────────────────────────────────────────
    student, dpt_head = load_model(args.student_checkpoint, device)

    # ── Forward pass ─────────────────────────────────────────────────────────
    print("\nRunning forward pass …")
    batch  = torch.stack(images)                          # (N, 3, H, W)
    preds  = predict(student, dpt_head, batch, device)    # (N, 1, H, W)

    # ── Per-sample stats ──────────────────────────────────────────────────────
    print(f"\n{'idx':>5}  {'GT min':>8}  {'GT max':>8}  {'GT mean':>8}  "
          f"{'Pred min':>9}  {'Pred max':>9}  {'Pred mean':>10}")
    print("─" * 68)
    for i, idx in enumerate(indices):
        gt = gt_depths[i].numpy()
        pd = preds[i, 0].numpy()
        print(f"{idx:>5}  "
              f"{gt.min():>8.3f}  {gt.max():>8.3f}  {gt[gt>0].mean():>8.3f}  "
              f"{pd.min():>9.3f}  {pd.max():>9.3f}  {pd.mean():>10.3f}")

    # ── Save figure ───────────────────────────────────────────────────────────
    save_figure(
        images    = images,
        gt_depths = gt_depths,
        pd_depths = [preds[i] for i in range(len(indices))],
        indices   = indices,
        out_path  = Path(args.output),
    )


if __name__ == "__main__":
    main()
