"""
batchsize_sweep.py — Effect of batch size on unfrozenBase_pretrainedUnfrozenHead.

Runs the best-performing configuration (unfrozen student + pretrained DPT head)
with batch sizes [1, 2, 4, 8] (powers of 2), each for --epochs epochs.
All other hyperparameters are identical across runs so batch size is the only
variable.

Each batch size gets its own W&B run (name: bs_1, bs_2, bs_4, bs_8) inside the
same project, making it easy to overlay the learning curves on one chart.

Checkpoints saved to:
    checkpoints/customization/batchsize_sweep/bs_1/
    checkpoints/customization/batchsize_sweep/bs_2/
    checkpoints/customization/batchsize_sweep/bs_4/
    checkpoints/customization/batchsize_sweep/bs_8/

Comparison chart saved to:
    checkpoints/customization/batchsize_sweep/comparison.png
    checkpoints/customization/batchsize_sweep/comparison.json

Usage:
    # Quick end-to-end test (30 epochs per batch size):
    python customization/batchsize_sweep.py --epochs 30

    # Full overnight run (1500 epochs per batch size):
    python customization/batchsize_sweep.py --epochs 1500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from customization.shared import (
    StudentWithDPTUnfrozen,
    run_training,
    get_fixed_val_indices,
)
from data.nyu_depth_dataset import NyuDepthDataset
from utils.device import get_device, pin_memory_for

# ImageNet normalisation constants (must match NyuDepthDataset)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_rgb(t: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalised (3,H,W) tensor to an (H,W,3) uint8 array."""
    img = (t.cpu() * _STD + _MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — hardcoded so the only argument needed is --epochs
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZES     = [1, 2, 4, 8, 16, 32, 64]          # powers of 2 to sweep
STUDENT_CKPT    = "checkpoints/best.pth"
NYU_ROOT        = "data/nyu_depth_v2"
IMG_SIZE        = 224
LR_STUDENT      = 1e-5   # same as the main unfrozenBase_pretrained experiment
LR_HEAD         = 1e-4
LR_MIN          = 1e-6
WEIGHT_DECAY    = 1e-2
GRAD_CLIP       = 1.0
AMP             = True
NUM_WORKERS     = 4
WANDB_PROJECT   = "efficient-distillation"
WANDB_ENTITY    = "ag11023-new-york-university"
WANDB_API_KEY   = "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"
SWEEP_DIR       = Path("checkpoints/customization/batchsize_sweep")

# Authenticate immediately so every wandb.init() call succeeds
os.environ["WANDB_API_KEY"] = WANDB_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing — only --epochs is exposed
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-size sweep for unfrozenBase_pretrainedUnfrozenHead")
    p.add_argument("--epochs", type=int, default=30,
                   help="Epochs per batch-size run  (default 30 for testing, use 1500 overnight)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Build dataloaders for a given batch size
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(batch_size: int, device: torch.device):
    """Return (train_loader, val_loader, val_dataset) for the given batch size."""
    pin = pin_memory_for(device)

    train_ds = NyuDepthDataset(root=NYU_ROOT, split="train", img_size=IMG_SIZE)
    val_ds   = NyuDepthDataset(root=NYU_ROOT, split="val",   img_size=IMG_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = pin,
        drop_last   = False,   # use every image every epoch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = max(batch_size, 8),  # val always runs at ≥8 for speed
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = pin,
        drop_last   = False,
    )
    return train_loader, val_loader, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper — collect predictions for a list of fixed samples
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model:          torch.nn.Module,
    fixed_samples:  list[dict],
    device:         torch.device,
    target_size:    tuple[int, int],
) -> list[torch.Tensor]:
    """
    Run inference on each sample in fixed_samples and return a list of
    (H, W) depth tensors on CPU — one per sample.

    target_size is (H, W) used to resize the output if the model returns
    a different spatial resolution.
    """
    model.eval()
    preds = []
    for sample in fixed_samples:
        x    = sample["images"].unsqueeze(0).to(device)   # (1, 3, H, W)
        pred = model(x).squeeze()                          # (H, W)
        if pred.shape != torch.Size(target_size):
            pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=target_size, mode="bicubic", align_corners=False,
            ).squeeze()
        preds.append(pred.cpu())
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# One sweep run — train for `epochs` with a given batch size
# ─────────────────────────────────────────────────────────────────────────────

def run_one_batch_size(
    batch_size:    int,
    epochs:        int,
    device:        torch.device,
    fixed_samples: list[dict],
    target_size:   tuple[int, int],
) -> dict:
    """
    Train unfrozenBase_pretrainedUnfrozenHead with `batch_size` for `epochs`.
    Returns the best metrics dict recorded during training.
    """
    run_name = f"bs_{batch_size}"
    save_dir = SWEEP_DIR / run_name

    print(f"\n{'='*60}")
    print(f"  BATCH SIZE = {batch_size}  (run: {run_name})")
    print(f"  Epochs: {epochs}  |  Save: {save_dir}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, val_dataset = build_loaders(batch_size, device)
    print(f"  Train batches/epoch : {len(train_loader):,}  "
          f"({len(train_loader.dataset):,} images, bs={batch_size})")
    print(f"  Val batches/epoch   : {len(val_loader):,}  "
          f"({len(val_loader.dataset):,} images)\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = StudentWithDPTUnfrozen(
        student_checkpoint = STUDENT_CKPT,
        dpt_pretrained     = True,    # pretrained Intel head (best config)
        freeze_student     = False,   # unfrozen student (best config)
    ).to(device)

    # ── Optimiser — same dual-LR setup as the main experiment ─────────────────
    optimiser = AdamW([
        {"params": model.student.parameters(), "lr": LR_STUDENT},
        {"params": model.dpt_head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=LR_MIN)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project = WANDB_PROJECT,
            entity  = WANDB_ENTITY,
            name    = run_name,
            group   = "batchsize_sweep",   # groups all 4 runs together in W&B
            config  = {
                "batch_size":   batch_size,
                "epochs":       epochs,
                "lr_student":   LR_STUDENT,
                "lr_head":      LR_HEAD,
                "lr_min":       LR_MIN,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip":    GRAD_CLIP,
                "amp":          AMP,
                "experiment":   "unfrozenBase_pretrainedUnfrozenHead",
            },
            resume = "allow",
            tags   = ["batch-sweep", f"bs-{batch_size}", "unfrozen-student", "pretrained-head"],
        )
        print(f"[wandb] {wandb_run.url}\n")
    except Exception as e:
        print(f"[wandb] Disabled ({e})\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    # Build a minimal args namespace with everything run_training expects
    args = SimpleNamespace(
        save_dir   = str(save_dir),
        epochs     = epochs,
        lr         = LR_HEAD,       # used only for W&B logging reference
        grad_clip  = GRAD_CLIP,
        amp        = AMP,
    )

    run_training(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        val_dataset  = val_dataset,
        optimiser    = optimiser,
        scheduler    = scheduler,
        device       = device,
        args         = args,
        run_name     = run_name,
        wandb_run    = wandb_run,
    )

    # ── Read best RMSE and reload best weights for prediction collection ─────────
    best_ckpt = save_dir / "best.pth"
    best_rmse = float("inf")
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        best_rmse = ckpt.get("best_rmse", float("inf"))
        # Reload the best weights so predictions come from the best checkpoint,
        # not the final epoch (which may be slightly worse)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Reloaded best.pth (RMSE {best_rmse:.4f}) for visual grid predictions")

    # ── Collect predictions on fixed 10 val images ────────────────────────────
    preds = collect_predictions(model, fixed_samples, device, target_size)

    return {"batch_size": batch_size, "best_rmse": best_rmse, "preds": preds}


# ─────────────────────────────────────────────────────────────────────────────
# Visual grid — 10 fixed images × (RGB | GT | bs_1 | bs_2 | bs_4 | bs_8)
# ─────────────────────────────────────────────────────────────────────────────

def save_visual_grid(
    results:       list[dict],   # each has "batch_size" and "preds" (list of H×W tensors)
    fixed_samples: list[dict],   # NyuDepthDataset samples for the 10 fixed images
    fixed_idxs:    list[int],
    out_path:      Path,
) -> None:
    """
    Grid figure: rows = fixed val images, columns = RGB | GT | bs_1 | bs_2 | bs_4 | bs_8.

    All depth maps in a row share the same colour scale (locked to GT range).
    Images are rotated 90° CCW so NYU frames appear upright.
    RMSE annotations are computed on the un-rotated arrays (rotation is display-only).
    """
    n_images = len(fixed_samples)
    bs_labels = [f"bs={r['batch_size']}" for r in results]
    col_labels = ["RGB Input", "Ground Truth"] + bs_labels
    n_cols     = len(col_labels)

    fig, axes = plt.subplots(n_images, n_cols,
                             figsize=(3 * n_cols, 3 * n_images))
    fig.suptitle(
        "Batch Size Sweep — unfrozenBase_pretrainedUnfrozenHead\n"
        "10 Fixed Val Images: RGB | GT | bs=1 | bs=2 | bs=4 | bs=8",
        fontsize=12, fontweight="bold", y=1.002,
    )

    # Column headers (top row only)
    for col, title in enumerate(col_labels):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold")

    for row, (sample, idx) in enumerate(zip(fixed_samples, fixed_idxs)):
        gt_np = sample["depths"].numpy()   # (H, W)
        valid = gt_np[gt_np > 0]
        vmin  = float(valid.min()) if len(valid) else 0.0
        vmax  = float(gt_np.max())

        # Rotate 90° CCW for display — metrics stay on unrotated arrays
        gt_disp = np.rot90(gt_np, k=1)

        # Column 0: RGB
        axes[row, 0].imshow(np.rot90(_to_rgb(sample["images"]), k=1))
        axes[row, 0].set_ylabel(f"idx {idx}", fontsize=7, rotation=0,
                                labelpad=40, va="center")
        axes[row, 0].axis("off")

        # Column 1: Ground truth
        axes[row, 1].imshow(gt_disp, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 1].axis("off")

        # Columns 2+: one per batch size
        for col, res in enumerate(results, start=2):
            pred_np = res["preds"][row].numpy()
            axes[row, col].imshow(np.rot90(pred_np, k=1), cmap="magma",
                                  vmin=vmin, vmax=vmax)
            # RMSE on unrotated arrays (rotation is purely cosmetic)
            mask = gt_np > 0
            rmse = float(np.sqrt(((pred_np[mask] - gt_np[mask]) ** 2).mean()))
            axes[row, col].set_xlabel(f"RMSE={rmse:.3f}m", fontsize=6)
            axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Visual grid saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison chart — produced after all batch sizes are done
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison(
    results:       list[dict],
    fixed_samples: list[dict],
    fixed_idxs:    list[int],
    out_dir:       Path,
) -> None:
    """
    Bar chart + JSON comparing best RMSE across all batch sizes.
    Also logs to W&B as a summary run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON (strip preds — not JSON-serialisable) ─────────────────────────
    json_path = out_dir / "comparison.json"
    json_results = [{"batch_size": r["batch_size"], "best_rmse": r["best_rmse"],
                     "elapsed_min": r.get("elapsed_min")} for r in results]
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nComparison JSON saved → {json_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    labels    = [f"bs={r['batch_size']}" for r in results]
    rmse_vals = [r["best_rmse"] for r in results]
    colors    = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"][:len(results)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, rmse_vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Batch size", fontsize=11)
    ax.set_ylabel("Best validation RMSE (↓ lower is better)", fontsize=10)
    ax.set_title("Batch Size Sweep — unfrozenBase_pretrainedUnfrozenHead\n"
                 "Best RMSE across all batch sizes", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rmse_vals) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(rmse_vals) * 1.15)
    plt.tight_layout()
    chart_path = out_dir / "comparison.png"
    fig.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison chart saved → {chart_path}")

    # ── Visual grid ───────────────────────────────────────────────────────────
    grid_path = out_dir / "visual_grid.png"
    save_visual_grid(results, fixed_samples, fixed_idxs, grid_path)

    # ── Log to W&B ────────────────────────────────────────────────────────────
    try:
        import wandb
        summary_run = wandb.init(
            project = WANDB_PROJECT,
            entity  = WANDB_ENTITY,
            name    = "batchsize_sweep_summary",
            group   = "batchsize_sweep",
            job_type= "summary",
            tags    = ["batch-sweep", "summary"],
        )
        # Table with one row per batch size
        table = wandb.Table(columns=["batch_size", "best_rmse"])
        for r in results:
            table.add_data(r["batch_size"], r["best_rmse"])
        summary_run.log({
            "sweep/best_rmse_table": table,
            "sweep/comparison_chart": wandb.Image(
                str(chart_path), caption="Best RMSE vs batch size"),
            "sweep/visual_grid": wandb.Image(
                str(grid_path),
                caption="Visual grid: RGB | GT | bs=1 | bs=2 | bs=4 | bs=8"),
        })
        for r in results:
            summary_run.summary[f"bs_{r['batch_size']}/best_rmse"] = r["best_rmse"]
        summary_run.finish()
        print("[wandb] Sweep summary logged (comparison chart + visual grid).")
    except Exception as e:
        print(f"[wandb] Summary disabled ({e})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    print("")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       Batch Size Sweep — unfrozenBase_pretrainedUnfrozenHead ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Batch sizes : {BATCH_SIZES}")
    print(f"║  Epochs each : {args.epochs}")
    print(f"║  Device      : {device}")
    print(f"║  W&B project : {WANDB_PROJECT}")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build the val dataset once and grab the 10 fixed images ───────────────
    val_ds      = NyuDepthDataset(root=NYU_ROOT, split="val", img_size=IMG_SIZE)
    fixed_idxs  = get_fixed_val_indices(len(val_ds))
    fixed_smpl  = [val_ds[i] for i in fixed_idxs]
    target_size = (IMG_SIZE, IMG_SIZE)
    print(f"  Fixed eval indices: {fixed_idxs}\n")

    results = []
    for bs in BATCH_SIZES:
        t0 = time.time()
        result = run_one_batch_size(bs, args.epochs, device, fixed_smpl, target_size)
        elapsed = time.time() - t0
        result["elapsed_min"] = round(elapsed / 60, 1)
        results.append(result)
        print(f"\n  ✓ bs={bs} done — best RMSE {result['best_rmse']:.4f}  "
              f"({result['elapsed_min']} min)\n")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("  BATCH SIZE SWEEP — SUMMARY")
    print("=" * 45)
    print(f"  {'Batch size':>12}  {'Best RMSE':>10}  {'Time (min)':>10}")
    print("  " + "-" * 38)
    for r in results:
        print(f"  {r['batch_size']:>12}  {r['best_rmse']:>10.4f}  {r['elapsed_min']:>10.1f}")
    print("=" * 45)

    best = min(results, key=lambda r: r["best_rmse"])
    print(f"\n  Best batch size: {best['batch_size']}  (RMSE {best['best_rmse']:.4f})")

    save_comparison(results, fixed_smpl, fixed_idxs, SWEEP_DIR)


if __name__ == "__main__":
    main()
