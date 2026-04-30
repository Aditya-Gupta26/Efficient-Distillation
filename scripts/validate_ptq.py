#!/usr/bin/env python3
"""
validate_ptq.py — Per-checkpoint validation + PTQ comparison on NYU-Depth V2.

For each .pth file in --checkpoints_dir:
  1. Validates checkpoint structure and model–dataset compatibility
  2. FP32 baseline validation  (RMSE, AbsRel, delta1, memory, latency)
  3. FP16 validation           (model.half() on GPU)
  4. Dynamic INT8 PTQ          (torch.quantization.quantize_dynamic, CPU)
  5. Saves all results to --output_dir/<timestamp>/{results.json, results.csv}
  6. Optionally logs metrics table + depth visualisations to W&B (--wandb)

Usage:
    python scripts/validate_ptq.py \\
        --checkpoints_dir checkpoints/customization \\
        --nyu_depth_root  /scratch/ag11023/HPML/nyu_depth_v2 \\
        --student_checkpoint checkpoints/epoch_069.pth \\
        --output_dir logs/ptq_validation \\
        --wandb --wandb_project depth-ptq

Quick test (2 batches per precision, skip INT8):
    python scripts/validate_ptq.py --max_val_batches 2 --skip_int8
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project root on sys.path ──────────────────────────────────────────────────
_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from data.nyu_depth_dataset import NyuDepthDataset
from models.depth_model import StudentWithDPT
from models.lora import apply_lora

try:
    import wandb as _wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

# ImageNet normalisation constants (must match training)
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested not in ("cpu", "cuda", "mps"):
        print(f"[warn] Unknown device '{requested}', falling back to cpu.")
    return torch.device("cpu")


def model_size_mb(model: nn.Module) -> float:
    mem = sum(p.nelement() * p.element_size() for p in model.parameters())
    mem += sum(b.nelement() * b.element_size() for b in model.buffers())
    return mem / 1024 ** 2


def disk_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024 ** 2


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Model tree printing
# ─────────────────────────────────────────────────────────────────────────────

def _build_tree(state: dict) -> dict:
    root: dict = {}
    for key, tensor in state.items():
        parts = key.split(".")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = tensor
    return root


def _count_node(node) -> int:
    if isinstance(node, torch.Tensor):
        return node.numel()
    return sum(_count_node(v) for v in node.values())


def _print_node(node, name="", prefix="", is_last=True, depth=0, max_depth=None):
    connector = "└── " if is_last else "├── "
    extension = "    " if is_last else "│   "

    if isinstance(node, torch.Tensor):
        dims = " × ".join(str(s) for s in node.shape) if node.shape else "scalar"
        print(f"{prefix}{connector}{name}: ({dims})  [{node.dtype}]")
        return

    n = _count_node(node)
    print(f"{prefix}{connector}{name}  [{n:,} params]" if name else f"{prefix}(root)  [{n:,} params]")

    if max_depth is not None and depth >= max_depth:
        print(f"{prefix}{extension}└── …")
        return

    children = list(node.items())
    for i, (child_name, child) in enumerate(children):
        _print_node(
            child,
            name      = child_name,
            prefix    = prefix + (extension if name else ""),
            is_last   = i == len(children) - 1,
            depth     = depth + 1,
            max_depth = max_depth,
        )


def print_model_tree(model: nn.Module, max_depth: int | None = None) -> None:
    state = model.state_dict()
    total = sum(t.numel() for t in state.values())
    print(f"\n  Built model — {total:,} total parameters")
    print(f"  {'─'*56}")
    tree     = _build_tree(state)
    children = list(tree.items())
    for i, (name, node) in enumerate(children):
        _print_node(
            node,
            name      = name,
            prefix    = "  ",
            is_last   = i == len(children) - 1,
            depth     = 0,
            max_depth = max_depth,
        )
    print(f"  {'─'*56}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_ckpt_type(ckpt: dict) -> str:
    """Return 'depth', 'lora_depth', 'distillation', or 'unknown'."""
    if "model" not in ckpt:
        if "student" in ckpt and "adapter" in ckpt:
            return "distillation"
        return "unknown"
    state = ckpt["model"]
    if any("lora_A" in k or "lora_B" in k for k in state.keys()):
        return "lora_depth"
    return "depth"


def infer_lora_rank(state_dict: dict) -> int:
    for k, v in state_dict.items():
        if "lora_A" in k:
            return int(v.shape[0])
    return 4


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    ckpt_path: str,
    student_ckpt: str,
    lora_alpha: float = 1.0,
) -> Tuple[nn.Module, str, dict]:
    raw       = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_type = detect_ckpt_type(raw)

    if ckpt_type not in ("depth", "lora_depth"):
        raise ValueError(
            f"Checkpoint {Path(ckpt_path).name!r} has type {ckpt_type!r}; "
            "only 'depth' and 'lora_depth' are supported here."
        )

    state   = raw["model"]
    is_lora = ckpt_type == "lora_depth"

    model = StudentWithDPT(
        student_checkpoint = student_ckpt,
        dpt_pretrained     = False,
        freeze_student     = True,
    )

    if is_lora:
        r = infer_lora_rank(state)
        for p in model.parameters():
            p.requires_grad = False
        apply_lora(model.student, r=r, alpha=lora_alpha, target_modules=["qkv"])
        print(f"  LoRA applied: rank={r}, alpha={lora_alpha}, scale={lora_alpha/r:.4f}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    non_head_missing    = [k for k in missing if "head" not in k]
    if non_head_missing:
        print(f"  [warn] {len(non_head_missing)} unexpected missing keys: "
              f"{non_head_missing[:3]}{'...' if len(non_head_missing) > 3 else ''}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys in checkpoint: {unexpected}")

    return model, ckpt_type, raw


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility check
# ─────────────────────────────────────────────────────────────────────────────

def check_compatibility(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[bool, str]:
    model.eval().to(device)
    try:
        batch  = next(iter(val_loader))
        images = batch["images"].to(device)

        with torch.no_grad():
            pred = model(images)

        B, _, H, W = images.shape
        pred_2d    = pred.squeeze(1)
        assert pred_2d.shape == (B, H, W), \
            f"Unexpected output shape {tuple(pred.shape)} — expected ({B}, [1,] {H}, {W})"
        assert not torch.isnan(pred).any(), "NaN values in model output"
        assert not torch.isinf(pred).any(), "Inf values in model output"
        assert (pred > 0).any(), "All predictions are non-positive"

        return True, f"OK — output {tuple(pred.shape)}, dtype {pred.dtype}"
    except Exception as exc:
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Depth metrics
# ─────────────────────────────────────────────────────────────────────────────

def _depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mask = (target > 0) & (pred > 1e-6)
    if mask.sum() == 0:
        return {"rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}
    p, t    = pred[mask], target[mask]
    rmse    = torch.sqrt(((p - t) ** 2).mean()).item()
    abs_rel = (torch.abs(p - t) / t).mean().item()
    ratio   = torch.max(p / t, t / p)
    delta1  = (ratio < 1.25     ).float().mean().item()
    delta2  = (ratio < 1.25 ** 2).float().mean().item()
    delta3  = (ratio < 1.25 ** 3).float().mean().item()
    return {"rmse": rmse, "abs_rel": abs_rel, "delta1": delta1, "delta2": delta2, "delta3": delta3}


# ─────────────────────────────────────────────────────────────────────────────
# Validation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(
    model:       nn.Module,
    val_loader:  DataLoader,
    device:      torch.device,
    half_inputs: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval().to(device)

    rmse_sum = abs_rel_sum = delta1_sum = delta2_sum = delta3_sum = 0.0
    latency_total = 0.0
    n = latency_n = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["images"].to(device)
        if half_inputs:
            images = images.half()
        depths = batch["depths"].to(device)

        t0   = time.perf_counter()
        pred = model(images).squeeze(1).float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        m = _depth_metrics(pred, depths)
        b = images.size(0)

        rmse_sum    += m["rmse"]    * b
        abs_rel_sum += m["abs_rel"] * b
        delta1_sum  += m["delta1"]  * b
        delta2_sum  += m["delta2"]  * b
        delta3_sum  += m["delta3"]  * b
        n           += b

        latency_total += dt
        latency_n     += b

    safe = lambda x: x / n if n else float("nan")
    return {
        "rmse":                   safe(rmse_sum),
        "abs_rel":                safe(abs_rel_sum),
        "delta1":                 safe(delta1_sum),
        "delta2":                 safe(delta2_sum),
        "delta3":                 safe(delta3_sum),
        "inference_ms_per_image": (latency_total / latency_n * 1000) if latency_n else float("nan"),
        "num_samples":            n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PTQ helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_dynamic_int8(model: nn.Module) -> nn.Module:
    cpu_model = copy.deepcopy(model).cpu().eval()
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic  # type: ignore[no-redef]
    return quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)


def make_fp16_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).half()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _denorm_rgb(tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor → (H,W,3) uint8."""
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)
    img = img * _IMG_STD + _IMG_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _colorize_depth(arr: np.ndarray) -> np.ndarray:
    """Float depth array → (H,W,3) uint8 using plasma colormap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        valid = arr[arr > 0]
        if valid.size == 0:
            return np.zeros((*arr.shape, 3), dtype=np.uint8)
        lo, hi = valid.min(), valid.max()
        norm   = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
        rgba   = cm.plasma(norm)
        return (rgba[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        gray = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


@torch.no_grad()
def collect_sample_preds(
    model:   nn.Module,
    dataset: NyuDepthDataset,
    indices: List[int],
    device:  torch.device,
    half:    bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Run model on the given dataset indices.
    Returns (rgb_list, gt_list, pred_list) — all numpy arrays, one per sample.
    rgb_list and gt_list are the same regardless of precision; returned here
    for convenience so the caller only needs one function call.
    """
    model.eval().to(device)
    rgbs, gts, preds = [], [], []
    for idx in indices:
        sample = dataset[idx]
        image  = sample["images"].unsqueeze(0).to(device)
        if half:
            image = image.half()
        pred = model(image).squeeze().cpu().float().numpy()
        rgbs.append(_denorm_rgb(sample["images"]))
        gts.append(sample["depths"].numpy())
        preds.append(pred)
    return rgbs, gts, preds


def make_vis_figure(
    rgbs:           List[np.ndarray],
    gts:            List[np.ndarray],
    preds_by_prec:  Dict[str, List[np.ndarray]],
) -> object:
    """
    Build a grid figure: rows = samples, cols = [RGB | GT | FP32 | FP16 | INT8 …]
    Returns a matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = len(rgbs)
    prec_keys = list(preds_by_prec.keys())
    n_cols    = 2 + len(prec_keys)

    col_titles = ["RGB Input", "Ground Truth"] + [p.upper().replace("_", " ") for p in prec_keys]

    fig, axes = plt.subplots(
        n_samples, n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_samples),
        squeeze=False,
    )

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row in range(n_samples):
        axes[row, 0].imshow(rgbs[row])
        axes[row, 1].imshow(_colorize_depth(gts[row]))
        for p_idx, prec in enumerate(prec_keys):
            axes[row, 2 + p_idx].imshow(_colorize_depth(preds_by_prec[prec][row]))
        for col in range(n_cols):
            axes[row, col].axis("off")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# W&B helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_wandb(args: argparse.Namespace):
    """Initialise a W&B run and return the run object, or None if unavailable."""
    if not args.wandb:
        return None
    if not _WANDB_OK:
        print("[warn] wandb not installed — skipping W&B logging. pip install wandb")
        return None
    run = _wandb.init(
        entity  = "ag11023-new-york-university",
        project = args.wandb_project,
        name    = args.wandb_run_name or None,
        config  = vars(args),
    )
    print(f"W&B run: {run.url}")
    return run


def log_to_wandb(
    run,
    ckpt_name:      str,
    result:         dict,
    vis_fig,
    out_dir:        Path,
) -> None:
    """Log metrics table + visualisation figure for one checkpoint."""
    if run is None or "precisions" not in result:
        return

    precs     = result["precisions"]
    fp32_rmse = precs.get("fp32", {}).get("rmse", float("nan"))

    # ── metrics table ─────────────────────────────────────────────────────────
    columns = ["Precision", "RMSE", "RMSE Δ vs FP32", "AbsRel", "δ1", "δ2", "δ3",
               "Mem (MB)", "Disk (MB)", "ms/img", "N Samples"]
    table   = _wandb.Table(columns=columns)

    for prec, m in precs.items():
        if m.get("skipped") or "error" in m:
            continue
        rmse  = m.get("rmse", float("nan"))
        delta = rmse - fp32_rmse if prec != "fp32" else 0.0
        table.add_data(
            prec,
            round(rmse,  4),
            round(delta, 4),
            round(m.get("abs_rel", float("nan")), 4),
            round(m.get("delta1",  float("nan")), 4),
            round(m.get("delta2",  float("nan")), 4),
            round(m.get("delta3",  float("nan")), 4),
            round(m.get("memory_mb", float("nan")), 1),
            round(m.get("disk_mb",   float("nan")), 1),
            round(m.get("inference_ms_per_image", float("nan")), 2),
            m.get("num_samples", 0),
        )

    log_dict: dict = {f"{ckpt_name}/metrics": table}

    # ── visualisation ─────────────────────────────────────────────────────────
    if vis_fig is not None:
        import matplotlib
        # Save to disk as well
        vis_path = out_dir / f"{ckpt_name}_depth_vis.png"
        vis_fig.savefig(vis_path, dpi=100, bbox_inches="tight")
        log_dict[f"{ckpt_name}/depth_predictions"] = _wandb.Image(str(vis_path))
        import matplotlib.pyplot as plt
        plt.close(vis_fig)

    run.log(log_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {json_path}")

    csv_path = output_dir / "results.csv"
    rows: List[dict] = []
    for r in results:
        base = {
            "checkpoint":    Path(r["checkpoint"]).name,
            "model_type":    r.get("model_type", ""),
            "ckpt_epoch":    r.get("ckpt_epoch", ""),
            "compatibility": r.get("compatibility", ""),
            "total_params":  r.get("total_params", ""),
        }
        if "precisions" not in r:
            rows.append({**base, "precision": "", "error": r.get("error", r.get("skip_reason", ""))})
            continue
        for prec, m in r["precisions"].items():
            rows.append({**base, "precision": prec, **m})

    if rows:
        fieldnames = list(rows[0].keys())
        for row in rows:
            for k in fieldnames:
                row.setdefault(k, "")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {csv_path}")


def print_precision_table(result: dict) -> None:
    """Print a per-checkpoint precision comparison table."""
    name  = Path(result["checkpoint"]).name
    precs = result.get("precisions", {})
    if not precs:
        return

    fp32_rmse = precs.get("fp32", {}).get("rmse", float("nan"))

    print(f"\n  ╔══ {name}  (epoch {result.get('ckpt_epoch','?')}) ══")
    print(f"  ║  {'Precision':<16} {'RMSE':>7} {'Δ FP32':>8} {'AbsRel':>8} "
          f"{'δ1':>7} {'δ2':>7} {'δ3':>7} {'Mem MB':>8} {'Disk MB':>8} {'ms/img':>8} {'N':>6}")
    print(f"  ║  {'─'*92}")

    for prec, m in precs.items():
        if "error" in m:
            print(f"  ║  {prec:<16}  ERROR: {m['error']}")
            continue
        if m.get("skipped"):
            print(f"  ║  {prec:<16}  SKIPPED: {m.get('reason','')}")
            continue

        rmse  = m.get("rmse",    float("nan"))
        delta = (rmse - fp32_rmse) if prec != "fp32" else 0.0
        sign  = "+" if delta > 0 else ""

        print(
            f"  ║  {prec:<16} "
            f"{rmse:>7.4f} "
            f"{sign+f'{delta:.4f}':>8} "
            f"{m.get('abs_rel', float('nan')):>8.4f} "
            f"{m.get('delta1',  float('nan')):>7.4f} "
            f"{m.get('delta2',  float('nan')):>7.4f} "
            f"{m.get('delta3',  float('nan')):>7.4f} "
            f"{m.get('memory_mb', float('nan')):>8.1f} "
            f"{m.get('disk_mb',   float('nan')):>8.1f} "
            f"{m.get('inference_ms_per_image', float('nan')):>8.2f} "
            f"{m.get('num_samples', 0):>6}"
        )
    print(f"  ╚{'═'*94}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoints_dir",    default="checkpoints/customization")
    p.add_argument("--nyu_depth_root",     default="/scratch/ag11023/HPML/nyu_depth_v2")
    p.add_argument("--student_checkpoint", default="checkpoints/epoch_069.pth")
    p.add_argument("--output_dir",         default="logs/ptq_validation")
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--img_size",           type=int,   default=224)
    p.add_argument("--device",             default="cuda")
    p.add_argument("--max_val_batches",    type=int,   default=None)
    p.add_argument("--int8_max_batches",   type=int,   default=100)
    p.add_argument("--lora_alpha",         type=float, default=1.0)
    p.add_argument("--skip_fp16",          action="store_true")
    p.add_argument("--skip_int8",          action="store_true")
    p.add_argument("--save_quantized",     action="store_true",
                   help="Save quantized model objects (FP16, INT8) to --output_dir.")
    # ── Visualisation ─────────────────────────────────────────────────────────
    p.add_argument("--n_vis",              type=int,   default=4,
                   help="Number of val samples to visualise per checkpoint.")
    p.add_argument("--vis_seed",           type=int,   default=42,
                   help="RNG seed for picking visualisation samples.")
    # ── W&B ───────────────────────────────────────────────────────────────────
    p.add_argument("--wandb",              action="store_true",
                   help="Enable W&B logging.")
    p.add_argument("--wandb_project",      default="depth-ptq-validation")
    p.add_argument("--wandb_run_name",     default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = _make_device(args.device)
    print(f"Primary device: {device}")

    # ── locate checkpoints ────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoints_dir)
    if not ckpt_dir.exists():
        print(f"[error] Checkpoints directory not found: {ckpt_dir}")
        sys.exit(1)
    ckpt_paths = sorted(ckpt_dir.rglob("*.pth"))
    if not ckpt_paths:
        print(f"[error] No .pth files found under {ckpt_dir}")
        sys.exit(1)
    print(f"Found {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {p.relative_to(ckpt_dir)}")

    # ── dataset ───────────────────────────────────────────────────────────────
    print(f"\nLoading NYU-Depth V2 val split from: {args.nyu_depth_root}")
    try:
        val_ds = NyuDepthDataset(
            root     = args.nyu_depth_root,
            split    = "val",
            img_size = args.img_size,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] Failed to load dataset: {exc}")
        sys.exit(1)

    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        drop_last   = False,
    )
    print(f"Val samples: {len(val_ds):,}  |  batches: {len(val_loader)}")

    # Pick fixed visualisation indices (same for every checkpoint)
    rng        = random.Random(args.vis_seed)
    vis_indices = rng.sample(range(len(val_ds)), min(args.n_vis, len(val_ds)))
    print(f"Vis sample indices (seed={args.vis_seed}): {vis_indices}")

    # ── output dir + W&B ──────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run  = init_wandb(args)
    all_results: List[dict] = []

    # ── process each checkpoint ───────────────────────────────────────────────
    for ckpt_path in ckpt_paths:
        print(f"\n{'─'*68}")
        print(f"[checkpoint] {ckpt_path.name}")

        result: dict = {"checkpoint": str(ckpt_path), "timestamp": ts}

        # Step 1 — detect type
        try:
            raw       = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ckpt_type = detect_ckpt_type(raw)
            result["model_type"] = ckpt_type
            result["ckpt_epoch"] = raw.get("epoch", "unknown")
            result["best_rmse"]  = raw.get("best_rmse", None)
            print(f"  Detected type: {ckpt_type}  |  epoch: {result['ckpt_epoch']}")

            if ckpt_type not in ("depth", "lora_depth"):
                result["compatibility"] = "skipped"
                result["skip_reason"]   = (
                    f"Type {ckpt_type!r} is not a depth checkpoint."
                )
                print(f"  [skip] {result['skip_reason']}")
                all_results.append(result)
                continue
        except Exception as exc:
            result.update(model_type="unknown", compatibility="failed",
                          error=f"Could not load checkpoint: {exc}")
            print(f"  [error] {result['error']}")
            all_results.append(result)
            continue

        # Step 2 — build model
        try:
            model, _, _ = load_model(
                ckpt_path    = str(ckpt_path),
                student_ckpt = args.student_checkpoint,
                lora_alpha   = args.lora_alpha,
            )
            result["total_params"] = count_params(model)
            print(f"  Total params: {result['total_params']:,}")
            print_model_tree(model, max_depth=2)
        except Exception as exc:
            result.update(compatibility="failed", error=f"Model build failed: {exc}")
            print(f"  [error] {result['error']}")
            all_results.append(result)
            continue

        # Step 3 — compatibility check
        compat_ok, compat_msg = check_compatibility(model, val_loader, device)
        result["compatibility"]        = "passed" if compat_ok else "failed"
        result["compatibility_detail"] = compat_msg
        print(f"  Compatibility: {result['compatibility']} — {compat_msg}")
        if not compat_ok:
            result["error"] = compat_msg
            all_results.append(result)
            continue

        result["precisions"] = {}
        ckpt_name        = ckpt_path.stem
        preds_by_prec: Dict[str, List[np.ndarray]] = {}
        vis_rgbs = vis_gts = None   # collected once from the fp32 pass

        # Step 4 — FP32
        print("\n  [FP32] Running validation ...")
        model_fp32 = copy.deepcopy(model).float().to(device)
        m_fp32     = run_validation(model_fp32, val_loader, device,
                                    half_inputs=False, max_batches=args.max_val_batches)
        result["precisions"]["fp32"] = {
            **m_fp32,
            "memory_mb": model_size_mb(model_fp32),
            "disk_mb":   disk_size_mb(model_fp32),
        }
        print(f"    RMSE={m_fp32['rmse']:.4f}  AbsRel={m_fp32['abs_rel']:.4f}  "
              f"δ1={m_fp32['delta1']:.4f}  δ2={m_fp32['delta2']:.4f}  δ3={m_fp32['delta3']:.4f}  "
              f"mem={result['precisions']['fp32']['memory_mb']:.1f}MB  "
              f"ms/img={m_fp32['inference_ms_per_image']:.2f}")

        vis_rgbs, vis_gts, fp32_preds = collect_sample_preds(
            model_fp32, val_ds, vis_indices, device, half=False)
        preds_by_prec["fp32"] = fp32_preds
        del model_fp32
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Step 5 — FP16
        if not args.skip_fp16:
            if device.type in ("cuda", "mps"):
                print("\n  [FP16] Running validation ...")
                model_fp16 = make_fp16_model(model).to(device)
                m_fp16     = run_validation(model_fp16, val_loader, device,
                                            half_inputs=True, max_batches=args.max_val_batches)
                result["precisions"]["fp16"] = {
                    **m_fp16,
                    "memory_mb": model_size_mb(model_fp16),
                    "disk_mb":   disk_size_mb(model_fp16),
                }
                print(f"    RMSE={m_fp16['rmse']:.4f}  AbsRel={m_fp16['abs_rel']:.4f}  "
                      f"δ1={m_fp16['delta1']:.4f}  δ2={m_fp16['delta2']:.4f}  δ3={m_fp16['delta3']:.4f}  "
                      f"mem={result['precisions']['fp16']['memory_mb']:.1f}MB  "
                      f"ms/img={m_fp16['inference_ms_per_image']:.2f}")

                _, _, fp16_preds = collect_sample_preds(
                    model_fp16, val_ds, vis_indices, device, half=True)
                preds_by_prec["fp16"] = fp16_preds

                if args.save_quantized:
                    fp16_path = out_dir / f"{ckpt_name}_fp16.pth"
                    torch.save(model_fp16, fp16_path)
                    result["precisions"]["fp16"]["saved_model"] = str(fp16_path)
                    print(f"  Saved FP16 model → {fp16_path}")

                del model_fp16
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                print("\n  [FP16] Skipped — requires CUDA or MPS.")
                result["precisions"]["fp16"] = {"skipped": True, "reason": "No GPU/MPS."}

        # Step 6 — INT8
        if not args.skip_int8:
            print(f"\n  [INT8-dynamic] Quantizing (CPU, max {args.int8_max_batches} batches) ...")
            try:
                model_int8 = apply_dynamic_int8(model)
                cpu        = torch.device("cpu")
                m_int8     = run_validation(model_int8, val_loader, cpu,
                                            half_inputs=False, max_batches=args.int8_max_batches)
                result["precisions"]["int8_dynamic"] = {
                    **m_int8,
                    "memory_mb": model_size_mb(model_int8),
                    "disk_mb":   disk_size_mb(model_int8),
                    "note": "Dynamic INT8 quantizes nn.Linear only; Conv2d stays FP32.",
                }
                m = result["precisions"]["int8_dynamic"]
                print(f"    RMSE={m_int8['rmse']:.4f}  AbsRel={m_int8['abs_rel']:.4f}  "
                      f"δ1={m_int8['delta1']:.4f}  δ2={m_int8['delta2']:.4f}  δ3={m_int8['delta3']:.4f}  "
                      f"mem={m['memory_mb']:.1f}MB  "
                      f"ms/img={m_int8['inference_ms_per_image']:.2f}")

                _, _, int8_preds = collect_sample_preds(
                    model_int8, val_ds, vis_indices, cpu, half=False)
                preds_by_prec["int8_dynamic"] = int8_preds

                if args.save_quantized:
                    int8_path = out_dir / f"{ckpt_name}_int8_dynamic.pth"
                    torch.save(model_int8, int8_path)
                    result["precisions"]["int8_dynamic"]["saved_model"] = str(int8_path)
                    print(f"  Saved INT8 model → {int8_path}")

                del model_int8
            except Exception as exc:
                print(f"  [warn] INT8 PTQ failed: {exc}")
                result["precisions"]["int8_dynamic"] = {"error": str(exc)}

        del model

        # Step 7 — per-checkpoint precision table
        print_precision_table(result)

        # Step 8 — visualisation + W&B logging
        vis_fig = None
        if vis_rgbs is not None and preds_by_prec:
            print(f"\n  Building depth visualisation ({len(vis_indices)} samples) ...")
            try:
                vis_fig = make_vis_figure(vis_rgbs, vis_gts, preds_by_prec)
                vis_path = out_dir / f"{ckpt_name}_depth_vis.png"
                vis_fig.savefig(vis_path, dpi=100, bbox_inches="tight")
                print(f"  Saved visualisation → {vis_path}")
            except Exception as exc:
                print(f"  [warn] Visualisation failed: {exc}")
                vis_fig = None

        log_to_wandb(wandb_run, ckpt_name, result, vis_fig, out_dir)

        if vis_fig is not None:
            import matplotlib.pyplot as plt
            plt.close(vis_fig)

        all_results.append(result)

    # ── final summary + save ──────────────────────────────────────────────────
    save_results(all_results, out_dir)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nAll done. Results: {out_dir}/")


if __name__ == "__main__":
    main()
