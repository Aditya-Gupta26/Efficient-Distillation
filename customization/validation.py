"""
validation.py — Side-by-side comparison of all trained models on NYU Depth V2 val.

Loads all 5 models (Intel DPT baseline + 4 trained variants), evaluates each
on the full val set, then produces:

  1. Metrics table  (RMSE, AbsRel, log-RMSE, SILog, δ1, δ2, δ3)
     Saved to:  checkpoints/customization/validation_metrics.json
                checkpoints/customization/validation_metrics.png   (bar chart)

  2. Visual grid   — the same 10 fixed images run through every model
     Columns: RGB | GT | Intel DPT | Exp1 | Exp2 | Exp3 | Exp4
     Saved to:  checkpoints/customization/validation_grid.png

Models evaluated:
  0. Intel/dpt-swinv2-tiny-256          (pretrained baseline — no training on NYU)
  1. frozenBase_pretrainedUnfrozenHead  (frozen student + pretrained head)
  2. unfrozenBase_pretrainedUnfrozenHead(unfrozen student + pretrained head)
  3. frozenBase_unfrozenHead            (frozen student + random head)
  4. unfrozenBase_unfrozenHead          (unfrozen student + random head)

Usage:
    python customization/validation.py \
        --student_checkpoint checkpoints/best.pth \
        --nyu_root data/nyu_depth_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from customization.shared import (
    compute_depth_metrics,
    get_fixed_val_indices,
    StudentWithDPTUnfrozen,
)
from models.depth_model import StudentWithDPT
from data.nyu_depth_dataset import NyuDepthDataset

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace local cache — Intel model downloaded once, loaded from disk after
# ─────────────────────────────────────────────────────────────────────────────
HF_CACHE_DIR = Path("checkpoints/hf_models/intel_dpt_swinv2_tiny_256")

# ─────────────────────────────────────────────────────────────────────────────
# Model registry — name, checkpoint path, model class, construction kwargs
# ─────────────────────────────────────────────────────────────────────────────

def get_model_registry(student_checkpoint: str) -> list[dict]:
    """
    Returns a list of model descriptors.  Each entry has:
      label      — short name for table / plot axes
      ckpt_path  — path to the best.pth for that experiment (None = Intel baseline)
      build_fn   — callable that returns an nn.Module ready for inference
    """
    BASE = Path("checkpoints/customization")

    return [
        {
            "label":    "Intel DPT\n(baseline)",
            "ckpt":     None,          # Intel model has no customization checkpoint
            "build":    lambda: _build_intel(),
            "color":    "#4e79a7",
        },
        {
            "label":    "Frozen student\nPretrained head\n(Exp 1)",
            "ckpt":     BASE / "frozenBase_pretrainedUnfrozenHead" / "best.pth",
            "build":    lambda: StudentWithDPT(
                            student_checkpoint=student_checkpoint,
                            dpt_pretrained=True, freeze_student=True),
            "color":    "#f28e2b",
        },
        {
            "label":    "Unfrozen student\nPretrained head\n(Exp 2)",
            "ckpt":     BASE / "unfrozenBase_pretrainedUnfrozenHead" / "best.pth",
            "build":    lambda: StudentWithDPTUnfrozen(
                            student_checkpoint=student_checkpoint,
                            dpt_pretrained=True, freeze_student=False),
            "color":    "#e15759",
        },
        {
            "label":    "Frozen student\nRandom head\n(Exp 3)",
            "ckpt":     BASE / "frozenBase_unfrozenHead" / "best.pth",
            "build":    lambda: StudentWithDPT(
                            student_checkpoint=student_checkpoint,
                            dpt_pretrained=False, freeze_student=True),
            "color":    "#76b7b2",
        },
        {
            "label":    "Unfrozen student\nRandom head\n(Exp 4)",
            "ckpt":     BASE / "unfrozenBase_unfrozenHead" / "best.pth",
            "build":    lambda: StudentWithDPTUnfrozen(
                            student_checkpoint=student_checkpoint,
                            dpt_pretrained=False, freeze_student=False),
            "color":    "#59a14f",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Intel DPT model builder (separate from our StudentWithDPT pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class _IntelDPTWrapper(torch.nn.Module):
    """Thin wrapper around HuggingFace DPTForDepthEstimation for uniform interface.

    Loads from local cache (HF_CACHE_DIR) if it exists; otherwise downloads
    from HuggingFace once and saves to the cache for future runs.
    """
    def __init__(self):
        super().__init__()
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        cache_dir = HF_CACHE_DIR.resolve()
        if cache_dir.exists() and any(cache_dir.iterdir()):
            print(f"  Loading Intel DPT from local cache: {cache_dir}")
            self._processor = DPTImageProcessor.from_pretrained(
                str(cache_dir), local_files_only=True)
            self._model = DPTForDepthEstimation.from_pretrained(
                str(cache_dir), local_files_only=True)
        else:
            print(f"  Downloading Intel/dpt-swinv2-tiny-256 from HuggingFace (one-time) …")
            self._processor = DPTImageProcessor.from_pretrained("Intel/dpt-swinv2-tiny-256")
            self._model     = DPTForDepthEstimation.from_pretrained("Intel/dpt-swinv2-tiny-256")
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._processor.save_pretrained(str(cache_dir))
            self._model.save_pretrained(str(cache_dir))
            print(f"  Model cached to {cache_dir}")
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) ImageNet-normalised tensor
        # We need PIL images for the processor — denormalise and convert
        from PIL import Image
        device = x.device
        results = []
        for img_t in x.cpu():
            arr = ((img_t * self._std + self._mean).clamp(0, 1)
                   .permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            inputs = {k: v.to(device) for k, v in
                      self._processor(images=pil, return_tensors="pt").items()}
            with torch.no_grad():
                pred = self._model(**inputs).predicted_depth   # (1, H, W)
            # Align scale to typical metre range using median
            results.append(pred.squeeze(0))
        depth = torch.stack(results, dim=0).unsqueeze(1)   # (B, 1, H, W)
        return depth


def _build_intel() -> _IntelDPTWrapper:
    print("Loading Intel/dpt-swinv2-tiny-256 …")
    return _IntelDPTWrapper()


# ─────────────────────────────────────────────────────────────────────────────
# Load a trained model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(entry: dict, device: torch.device):
    """
    Build the model and load its best.pth checkpoint.
    Returns (model, available) where available=False if no checkpoint exists.
    """
    model = entry["build"]()

    if entry["ckpt"] is None:
        # Intel baseline — already loaded with pretrained weights
        model.to(device).eval()
        return model, True

    ckpt_path = Path(entry["ckpt"])
    if not ckpt_path.exists():
        print(f"  [skip] {entry['label'].replace(chr(10),' ')} — checkpoint not found: {ckpt_path}")
        return None, False

    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    best_rmse = ckpt.get("best_rmse", "?")
    print(f"  Loaded: {ckpt_path.parent.name}  (best RMSE {best_rmse:.4f})")
    return model, True


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: torch.device,
            target_size: tuple[int, int]) -> torch.Tensor:
    """
    Run a single (1,3,H,W) image through model, return (H,W) depth on CPU.
    For the Intel wrapper the output may need scale alignment.
    """
    x    = image_tensor.unsqueeze(0).to(device)
    pred = model(x).squeeze()   # (H, W) or (H, W) after squeeze

    # Resize if model output differs from target (e.g. Intel 256×256 → 224×224)
    if pred.shape != torch.Size(target_size):
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0),
                             size=target_size, mode="bicubic",
                             align_corners=False).squeeze()
    return pred.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics table
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_full_val(model, dataset: NyuDepthDataset, device: torch.device,
                      label: str) -> dict[str, float]:
    """Run model on all val images and return averaged metrics."""
    sums = {k: 0.0 for k in ["rmse", "abs_rel", "log_rmse", "silog",
                               "delta1", "delta2", "delta3"]}
    n = 0
    target_size = (dataset.img_size, dataset.img_size)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        pred   = predict(model, sample["images"], device, target_size)
        gt     = sample["depths"]

        # Scale-align Intel predictions (relative → metric)
        if isinstance(model, _IntelDPTWrapper):
            gt_valid   = gt[gt > 0]
            pred_valid = pred[gt > 0]
            if gt_valid.numel() > 0 and pred_valid.median() > 1e-6:
                pred = pred * (gt_valid.median() / pred_valid.median())

        m = compute_depth_metrics(pred, gt)
        for k in sums:
            sums[k] += m[k]
        n += 1

        if (idx + 1) % 30 == 0:
            print(f"    {label[:25]:25s}  [{idx+1}/{len(dataset)}]  "
                  f"RMSE={sums['rmse']/n:.4f}", end="\r", flush=True)

    print()
    return {k: v / n for k, v in sums.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_rgb(t: torch.Tensor) -> np.ndarray:
    img = (t.cpu() * _STD + _MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_metrics_chart(results: list[dict], out_path: Path) -> None:
    """Bar chart comparing all models on every metric."""
    metrics   = ["rmse", "abs_rel", "log_rmse", "silog", "delta1", "delta2", "delta3"]
    labels    = [r["label"].replace("\n", " ") for r in results]
    colors    = [r["color"] for r in results]
    n_metrics = len(metrics)
    n_models  = len(results)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    fig.suptitle("NYU Depth V2 Val — Model Comparison", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        vals = [r["metrics"][metric] for r in results]
        bars = ax.bar(range(n_models), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(metric.upper(), fontsize=10, fontweight="bold")
        # Annotate bar values
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        # For delta metrics higher is better — note on chart
        if metric.startswith("delta"):
            ax.set_ylabel("↑ higher is better", fontsize=7)
        else:
            ax.set_ylabel("↓ lower is better", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics chart saved → {out_path}")


def save_visual_grid(
    results:  list[dict],      # each has "label", "preds" (list of H×W tensors)
    samples:  list[dict],      # NyuDepthDataset samples (fixed 10 images)
    indices:  list[int],
    out_path: Path,
) -> None:
    """
    Grid figure: rows = images, columns = RGB | GT | model1 | model2 | …

    Each cell shows the depth map.  All depth maps in a row share the same
    colour scale (locked to the ground-truth range) for fair visual comparison.
    """
    n_images  = len(samples)
    # Columns: RGB + GT + one per model
    col_labels = ["RGB Input", "Ground Truth"] + [r["label"] for r in results]
    n_cols     = len(col_labels)

    fig, axes = plt.subplots(n_images, n_cols,
                             figsize=(3 * n_cols, 3 * n_images))
    fig.suptitle("NYU Depth V2 — Visual Comparison Across All Models",
                 fontsize=13, fontweight="bold", y=1.002)

    # Column headers (only top row)
    for col, title in enumerate(col_labels):
        axes[0, col].set_title(title.replace("\n", "\n"),
                               fontsize=8, fontweight="bold")

    for row, (sample, idx) in enumerate(zip(samples, indices)):
        gt_np = sample["depths"].numpy()
        valid = gt_np[gt_np > 0]
        vmin  = float(valid.min()) if len(valid) else 0.0
        vmax  = float(gt_np.max())

        # Rotate 90° counter-clockwise for display so NYU images appear upright.
        # np.rot90(arr, k=1) rotates CCW once.  Metrics use the unrotated arrays.
        gt_disp = np.rot90(gt_np, k=1)

        # Column 0: RGB
        axes[row, 0].imshow(np.rot90(_to_rgb(sample["images"]), k=1))
        axes[row, 0].set_ylabel(f"idx {idx}", fontsize=7, rotation=0,
                                labelpad=40, va="center")
        axes[row, 0].axis("off")

        # Column 1: Ground truth
        axes[row, 1].imshow(gt_disp, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 1].axis("off")

        # Remaining columns: model predictions
        for col, res in enumerate(results, start=2):
            pred_np = res["preds"][row].numpy()
            axes[row, col].imshow(np.rot90(pred_np, k=1), cmap="magma", vmin=vmin, vmax=vmax)
            # RMSE computed on unrotated arrays (rotation doesn't change pixel values)
            mask = gt_np > 0
            rmse = float(np.sqrt(((pred_np[mask] - gt_np[mask]) ** 2).mean()))
            axes[row, col].set_xlabel(f"RMSE={rmse:.3f}m", fontsize=6)
            axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Visual grid saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare all models on NYU Depth V2 val")
    p.add_argument("--student_checkpoint", default="checkpoints/best.pth")
    p.add_argument("--nyu_root",      default="data/nyu_depth_v2")
    p.add_argument("--img_size",      type=int, default=224)
    p.add_argument("--out_dir",       default="checkpoints/customization")
    p.add_argument("--wandb_project", default="efficient-distillation",
                   help="W&B project to log final comparison results into")
    p.add_argument("--wandb_entity",  default=None,
                   help="W&B entity (username or team); omit to use default")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}\n")

    # ── W&B — log final comparison table, metrics chart, and visual grid ──────
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = "validation_comparison",
            job_type= "evaluation",
            tags    = ["validation", "comparison", "all-models"],
        )
        print(f"[wandb] {wandb_run.url}\n")
    except Exception as e:
        print(f"[wandb] Disabled ({e})\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset     = NyuDepthDataset(root=args.nyu_root, split="val", img_size=args.img_size)
    fixed_idxs  = get_fixed_val_indices(len(dataset))
    fixed_smpl  = [dataset[i] for i in fixed_idxs]
    target_size = (args.img_size, args.img_size)
    print(f"Val split: {len(dataset)} images  |  Fixed eval set: {fixed_idxs}\n")

    # ── Load all models and evaluate ──────────────────────────────────────────
    registry = get_model_registry(args.student_checkpoint)
    results  = []

    for entry in registry:
        lbl = entry["label"].replace("\n", " ")
        print(f"─── {lbl} ───")
        model, available = load_model(entry, device)
        if not available:
            continue

        # Full val metrics
        print(f"  Evaluating on {len(dataset)} val images …")
        metrics = evaluate_full_val(model, dataset, device, lbl)

        # Predictions on fixed 10 images for visual grid
        preds = []
        for sample in fixed_smpl:
            pred = predict(model, sample["images"], device, target_size)
            if isinstance(model, _IntelDPTWrapper):
                gt_valid = sample["depths"][sample["depths"] > 0]
                pv = pred[sample["depths"] > 0]
                if gt_valid.numel() > 0 and pv.median() > 1e-6:
                    pred = pred * (gt_valid.median() / pv.median())
            preds.append(pred)

        results.append({
            "label":   entry["label"],
            "color":   entry["color"],
            "metrics": metrics,
            "preds":   preds,
        })

        # Print metrics for this model
        m = metrics
        print(f"  RMSE={m['rmse']:.4f}  AbsRel={m['abs_rel']:.4f}  "
              f"SILog={m['silog']:.4f}  δ1={m['delta1']:.4f}\n")

    # ── Print summary table ───────────────────────────────────────────────────
    metric_keys = ["rmse", "abs_rel", "log_rmse", "silog", "delta1", "delta2", "delta3"]
    col_w = 12
    header = f"{'Model':<35}" + "".join(f"{k:>{col_w}}" for k in metric_keys)
    print("\n" + "=" * len(header))
    print("  FINAL METRICS COMPARISON — NYU Depth V2 Validation Set")
    print("=" * len(header))
    print(header)
    print("─" * len(header))
    for res in results:
        name = res["label"].replace("\n", " ")
        row  = f"{name:<35}" + "".join(f"{res['metrics'][k]:>{col_w}.4f}" for k in metric_keys)
        print(row)
    print("=" * len(header))
    print("  ↓ lower is better for: RMSE, AbsRel, log-RMSE, SILog")
    print("  ↑ higher is better for: δ1, δ2, δ3")

    # ── Save JSON results ─────────────────────────────────────────────────────
    json_out = out_dir / "validation_metrics.json"
    with open(json_out, "w") as f:
        json.dump([{"label": r["label"].replace("\n"," "), "metrics": r["metrics"]}
                   for r in results], f, indent=2)
    print(f"\nMetrics saved → {json_out}")

    # ── Save metrics bar chart ────────────────────────────────────────────────
    chart_path = out_dir / "validation_metrics.png"
    save_metrics_chart(results, chart_path)

    # ── Save visual grid ──────────────────────────────────────────────────────
    grid_path = out_dir / "validation_grid.png"
    save_visual_grid(results, fixed_smpl, fixed_idxs, grid_path)

    # ── Log everything to W&B ─────────────────────────────────────────────────
    if wandb_run is not None:
        import wandb

        # 1. Comparison table — every model × every metric in one structured view
        metric_keys = ["rmse", "abs_rel", "log_rmse", "silog", "delta1", "delta2", "delta3"]
        table = wandb.Table(columns=["model"] + metric_keys)
        for res in results:
            row = [res["label"].replace("\n", " ")] + [res["metrics"][k] for k in metric_keys]
            table.add_data(*row)
        wandb_run.log({"comparison/metrics_table": table})

        # 2. Scalar summary — makes it easy to sort runs by a single metric
        for res in results:
            safe_name = res["label"].replace("\n", "_").replace(" ", "_").replace("(","").replace(")","")
            for k, v in res["metrics"].items():
                wandb_run.summary[f"{safe_name}/{k}"] = v

        # 3. Metrics bar chart and visual grid as images
        wandb_run.log({
            "comparison/metrics_chart": wandb.Image(str(chart_path),
                caption="Bar chart: all models vs all metrics"),
            "comparison/visual_grid":   wandb.Image(str(grid_path),
                caption="Visual comparison: RGB | GT | Intel DPT | Exp1 | Exp2 | Exp3 | Exp4"),
        })

        wandb_run.finish()
        print(f"\n[wandb] Results logged to project '{args.wandb_project}' → run 'validation_comparison'")


if __name__ == "__main__":
    main()
