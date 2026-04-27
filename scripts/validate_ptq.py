#!/usr/bin/env python3
"""
validate_ptq.py — Per-checkpoint validation + PTQ comparison on NYU-Depth V2.

For each .pth file in --checkpoints_dir:
  1. Validates checkpoint structure and model–dataset compatibility
  2. FP32 baseline validation  (RMSE, AbsRel, delta1, memory, latency)
  3. FP16 validation           (model.half() on GPU)
  4. Dynamic INT8 PTQ          (torch.quantization.quantize_dynamic, CPU)
  5. Saves all results to --output_dir/<timestamp>/{results.json, results.csv}

Usage:
    python scripts/validate_ptq.py \\
        --checkpoints_dir checkpoints/customization \\
        --nyu_depth_root  /scratch/ag11023/HPML/nyu_depth_v2 \\
        --student_checkpoint checkpoints/epoch_069.pth \\
        --output_dir logs/ptq_validation

Quick test (2 batches per precision, skip INT8):
    python scripts/validate_ptq.py --max_val_batches 2 --skip_int8
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project root on sys.path ──────────────────────────────────────────────────
_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from data.nyu_depth_dataset import NyuDepthDataset
from models.depth_model import StudentWithDPT
from models.lora import apply_lora


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
    """Actual in-memory size of parameters + buffers in MB."""
    mem = sum(p.nelement() * p.element_size() for p in model.parameters())
    mem += sum(b.nelement() * b.element_size() for b in model.buffers())
    return mem / 1024 ** 2


def disk_size_mb(model: nn.Module) -> float:
    """State-dict serialized size in MB (proxy for .pth file size)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024 ** 2


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_ckpt_type(ckpt: dict) -> str:
    """Return 'depth', 'lora_depth', 'distillation', or 'unknown'."""
    if "model" not in ckpt:
        if "student" in ckpt and "adapter" in ckpt:
            return "distillation"
        return "unknown"
    # Has a "model" key → depth or lora_depth
    state = ckpt["model"]
    if any("lora_A" in k or "lora_B" in k for k in state.keys()):
        return "lora_depth"
    return "depth"


def infer_lora_rank(state_dict: dict) -> int:
    """Infer LoRA rank from the shape of the first lora_A tensor (r, in_features)."""
    for k, v in state_dict.items():
        if "lora_A" in k:
            return int(v.shape[0])
    return 4  # default from lora_depth_config.yaml


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    ckpt_path: str,
    student_ckpt: str,
    lora_alpha: float = 1.0,
) -> Tuple[nn.Module, str, dict]:
    """
    Load StudentWithDPT from a depth/lora_depth checkpoint.

    Returns (model, ckpt_type, raw_ckpt_dict).

    Loading order (matches train.py):
      1. Build StudentWithDPT with student_checkpoint (architecture only;
         weights will be overwritten by the full model state dict).
      2. For lora_depth: freeze all params, then apply LoRA adapters so that
         the state dict keys match (LoRALinear vs nn.Linear).
      3. Load the full "model" state dict from the customization checkpoint.
    """
    raw       = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_type = detect_ckpt_type(raw)

    if ckpt_type not in ("depth", "lora_depth"):
        raise ValueError(
            f"Checkpoint {Path(ckpt_path).name!r} has type {ckpt_type!r}; "
            "only 'depth' and 'lora_depth' are supported here."
        )

    state    = raw["model"]
    is_lora  = ckpt_type == "lora_depth"

    # dpt_pretrained=False avoids a HuggingFace network download; weights
    # from the customization checkpoint will overwrite everything anyway.
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
        print(f"  [warn] {len(unexpected)} unexpected keys in checkpoint.")

    return model, ckpt_type, raw


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility check
# ─────────────────────────────────────────────────────────────────────────────

def check_compatibility(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[bool, str]:
    """
    Run one forward pass to verify model/dataset compatibility.
    Returns (passed: bool, message: str).
    """
    model.eval().to(device)
    try:
        batch  = next(iter(val_loader))
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)

        with torch.no_grad():
            pred = model(images)

        # Accept (B, 1, H, W) or (B, H, W) — custom heads may already squeeze
        B, _, H, W   = images.shape
        pred_2d      = pred.squeeze(1)   # no-op if already (B, H, W)
        assert pred_2d.shape == (B, H, W), \
            f"Unexpected output shape {tuple(pred.shape)} — expected ({B}, [1,] {H}, {W})"
        assert not torch.isnan(pred).any(),  "NaN values in model output"
        assert not torch.isinf(pred).any(),  "Inf values in model output"
        assert (pred > 0).any(), "All predictions are non-positive"

        return True, f"OK — output {tuple(pred.shape)}, dtype {pred.dtype}"
    except Exception as exc:
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Depth metrics (mirrors DepthTrainer._depth_metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mask = (target > 0) & (pred > 1e-6)
    if mask.sum() == 0:
        return {"rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}
    p, t    = pred[mask], target[mask]
    rmse    = torch.sqrt(((p - t) ** 2).mean()).item()
    abs_rel = (torch.abs(p - t) / t).mean().item()
    delta1  = (torch.max(p / t, t / p) < 1.25).float().mean().item()
    return {"rmse": rmse, "abs_rel": abs_rel, "delta1": delta1}


# ─────────────────────────────────────────────────────────────────────────────
# Validation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(
    model:        nn.Module,
    val_loader:   DataLoader,
    device:       torch.device,
    half_inputs:  bool = False,   # True for FP16 model
    max_batches:  Optional[int] = None,
) -> Dict[str, float]:
    """
    Returns {rmse, abs_rel, delta1, inference_ms_per_image, num_samples}.
    half_inputs=True casts images to float16 (for a FP16 model).
    """
    model.eval().to(device)

    rmse_sum = abs_rel_sum = delta1_sum = 0.0
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
        pred = model(images).squeeze(1).float()  # always float32 for metrics
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt   = time.perf_counter() - t0

        m = _depth_metrics(pred, depths)
        b = images.size(0)

        rmse_sum    += m["rmse"]    * b
        abs_rel_sum += m["abs_rel"] * b
        delta1_sum  += m["delta1"]  * b
        n           += b

        latency_total += dt
        latency_n     += b

    safe = lambda x: x / n if n else float("nan")
    return {
        "rmse":                    safe(rmse_sum),
        "abs_rel":                 safe(abs_rel_sum),
        "delta1":                  safe(delta1_sum),
        "inference_ms_per_image":  (latency_total / latency_n * 1000) if latency_n else float("nan"),
        "num_samples":             n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PTQ helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_dynamic_int8(model: nn.Module) -> nn.Module:
    """
    Dynamic INT8 quantization targeting nn.Linear modules (CPU only).
    Note: LoRALinear is a custom module, so only standard nn.Linear layers
    (primarily in the DPT head) are quantized here.
    """
    cpu_model = copy.deepcopy(model).cpu().eval()
    # torch.ao.quantization is the future-proof API; fall back to legacy
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic  # type: ignore[no-redef]

    return quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)


def make_fp16_model(model: nn.Module) -> nn.Module:
    """Deep-copy the model and cast all parameters/buffers to float16."""
    return copy.deepcopy(model).half()


# ─────────────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {json_path}")

    # Flatten to one row per (checkpoint × precision) for CSV
    csv_path = output_dir / "results.csv"
    rows: List[dict] = []
    for r in results:
        base = {
            "checkpoint":   Path(r["checkpoint"]).name,
            "model_type":   r.get("model_type", ""),
            "ckpt_epoch":   r.get("ckpt_epoch", ""),
            "compatibility": r.get("compatibility", ""),
            "total_params": r.get("total_params", ""),
        }
        if "precisions" not in r:
            rows.append({**base, "precision": "", "error": r.get("error", r.get("skip_reason", ""))})
            continue
        for prec, m in r["precisions"].items():
            rows.append({**base, "precision": prec, **m})

    if rows:
        fieldnames = list(rows[0].keys())
        # Ensure all rows have all keys
        for row in rows:
            for k in fieldnames:
                row.setdefault(k, "")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {csv_path}")


def print_summary(results: List[dict]) -> None:
    for r in results:
        name = Path(r["checkpoint"]).name
        print(f"\n{'='*68}")
        print(f"  {name}  |  type: {r.get('model_type','?')}  |  epoch: {r.get('ckpt_epoch','?')}")
        print(f"  Compatibility: {r.get('compatibility','?')}")
        if "error" in r:
            print(f"  Error: {r['error']}")
            continue
        if "skip_reason" in r:
            print(f"  Skipped: {r['skip_reason']}")
            continue

        total = r.get("total_params")
        if total:
            print(f"  Total params: {total:,}")

        precs = r.get("precisions", {})
        if not precs:
            continue

        # Header
        print(f"\n  {'Precision':<14} {'RMSE':>7} {'AbsRel':>8} {'delta1':>8} "
              f"{'Mem(MB)':>9} {'Disk(MB)':>9} {'ms/img':>8} {'N':>6}")
        print(f"  {'-'*72}")

        fp32_rmse = precs.get("fp32", {}).get("rmse", float("nan"))

        for prec, m in precs.items():
            if "error" in m:
                print(f"  {prec:<14}  ERROR: {m['error']}")
                continue
            if m.get("skipped"):
                print(f"  {prec:<14}  SKIPPED: {m.get('reason','')}")
                continue

            rmse   = m.get("rmse",    float("nan"))
            delta  = (rmse - fp32_rmse) if prec != "fp32" else 0.0
            sign   = "+" if delta >= 0 else ""

            print(
                f"  {prec:<14} "
                f"{rmse:>7.4f} "
                f"{m.get('abs_rel', float('nan')):>8.4f} "
                f"{m.get('delta1',  float('nan')):>8.4f} "
                f"{m.get('memory_mb', float('nan')):>9.1f} "
                f"{m.get('disk_mb',   float('nan')):>9.1f} "
                f"{m.get('inference_ms_per_image', float('nan')):>8.2f} "
                f"{m.get('num_samples', 0):>6}"
                + (f"  ({sign}{delta:.4f} vs fp32)" if prec != "fp32" else "")
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoints_dir",    default="checkpoints/customization",
                   help="Directory of .pth checkpoints to evaluate.")
    p.add_argument("--nyu_depth_root",     default="/scratch/ag11023/HPML/nyu_depth_v2",
                   help="Path to NYU-Depth V2 dataset root.")
    p.add_argument("--student_checkpoint", default="checkpoints/epoch_069.pth",
                   help="Base distillation checkpoint (defines Swin-Tiny architecture).")
    p.add_argument("--output_dir",         default="logs/ptq_validation",
                   help="Directory for results.json and results.csv.")
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--img_size",           type=int,   default=224)
    p.add_argument("--device",             default="cuda",
                   help="Device for FP32/FP16 runs. INT8 always runs on CPU.")
    p.add_argument("--max_val_batches",    type=int,   default=None,
                   help="Cap FP32/FP16 validation at N batches (useful for smoke tests).")
    p.add_argument("--int8_max_batches",   type=int,   default=100,
                   help="Cap INT8 validation at N batches (CPU is slow).")
    p.add_argument("--lora_alpha",         type=float, default=1.0,
                   help="LoRA alpha used during training (needed to reconstruct scale).")
    p.add_argument("--skip_fp16",          action="store_true",
                   help="Skip FP16 evaluation.")
    p.add_argument("--skip_int8",          action="store_true",
                   help="Skip INT8 dynamic PTQ evaluation.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = _make_device(args.device)
    print(f"Primary device: {device}")

    # ── locate checkpoints ────────────────────────────────────────────────────
    ckpt_dir   = Path(args.checkpoints_dir)
    if not ckpt_dir.exists():
        print(f"[error] Checkpoints directory not found: {ckpt_dir}")
        sys.exit(1)
    ckpt_paths = sorted(ckpt_dir.rglob("*.pth"))
    if not ckpt_paths:
        print(f"[error] No .pth files found under {ckpt_dir} (searched recursively)")
        sys.exit(1)
    print(f"Found {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {p.relative_to(ckpt_dir)}")

    # ── build validation dataloader ───────────────────────────────────────────
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

    # ── output directory ──────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts

    all_results: List[dict] = []

    # ── process each checkpoint ───────────────────────────────────────────────
    for ckpt_path in ckpt_paths:
        print(f"\n{'─'*68}")
        print(f"[checkpoint] {ckpt_path.name}")

        result: dict = {
            "checkpoint": str(ckpt_path),
            "timestamp":  ts,
        }

        # Step 1 — load raw checkpoint and detect type
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
                    f"Type {ckpt_type!r} is not a depth model checkpoint. "
                    "Only 'depth' and 'lora_depth' checkpoints are evaluated."
                )
                print(f"  [skip] {result['skip_reason']}")
                all_results.append(result)
                continue

        except Exception as exc:
            result["model_type"]    = "unknown"
            result["compatibility"] = "failed"
            result["error"]         = f"Could not load checkpoint: {exc}"
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
        except Exception as exc:
            result["compatibility"] = "failed"
            result["error"]         = f"Model build failed: {exc}"
            print(f"  [error] {result['error']}")
            all_results.append(result)
            continue

        # Step 3 — compatibility check (single forward pass)
        compat_ok, compat_msg = check_compatibility(model, val_loader, device)
        result["compatibility"]        = "passed" if compat_ok else "failed"
        result["compatibility_detail"] = compat_msg
        print(f"  Compatibility: {result['compatibility']} — {compat_msg}")

        if not compat_ok:
            result["error"] = compat_msg
            all_results.append(result)
            continue

        result["precisions"] = {}

        # Step 4 — FP32 baseline validation
        print("\n  [FP32] Running validation ...")
        model_fp32 = copy.deepcopy(model).float().to(device)
        m_fp32     = run_validation(
            model_fp32, val_loader, device,
            half_inputs=False, max_batches=args.max_val_batches,
        )
        result["precisions"]["fp32"] = {
            **m_fp32,
            "memory_mb": model_size_mb(model_fp32),
            "disk_mb":   disk_size_mb(model_fp32),
        }
        print(f"    RMSE={m_fp32['rmse']:.4f}  AbsRel={m_fp32['abs_rel']:.4f}  "
              f"delta1={m_fp32['delta1']:.4f}  "
              f"mem={result['precisions']['fp32']['memory_mb']:.1f}MB  "
              f"ms/img={m_fp32['inference_ms_per_image']:.2f}")
        del model_fp32
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Step 5 — FP16 validation (GPU / MPS only)
        if not args.skip_fp16:
            if device.type in ("cuda", "mps"):
                print("\n  [FP16] Running validation (model.half()) ...")
                model_fp16 = make_fp16_model(model).to(device)
                m_fp16     = run_validation(
                    model_fp16, val_loader, device,
                    half_inputs=True, max_batches=args.max_val_batches,
                )
                result["precisions"]["fp16"] = {
                    **m_fp16,
                    "memory_mb": model_size_mb(model_fp16),
                    "disk_mb":   disk_size_mb(model_fp16),
                }
                print(f"    RMSE={m_fp16['rmse']:.4f}  AbsRel={m_fp16['abs_rel']:.4f}  "
                      f"delta1={m_fp16['delta1']:.4f}  "
                      f"mem={result['precisions']['fp16']['memory_mb']:.1f}MB  "
                      f"ms/img={m_fp16['inference_ms_per_image']:.2f}")
                del model_fp16
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                print("\n  [FP16] Skipped — requires CUDA or MPS device.")
                result["precisions"]["fp16"] = {
                    "skipped": True,
                    "reason":  "No GPU/MPS device available.",
                }

        # Step 6 — Dynamic INT8 PTQ (CPU)
        if not args.skip_int8:
            print(f"\n  [INT8-dynamic] Quantizing and running on CPU "
                  f"(max {args.int8_max_batches} batches) ...")
            try:
                model_int8 = apply_dynamic_int8(model)
                cpu        = torch.device("cpu")
                m_int8     = run_validation(
                    model_int8, val_loader, cpu,
                    half_inputs=False, max_batches=args.int8_max_batches,
                )
                result["precisions"]["int8_dynamic"] = {
                    **m_int8,
                    "memory_mb":  model_size_mb(model_int8),
                    "disk_mb":    disk_size_mb(model_int8),
                    "note": (
                        "Dynamic INT8 quantizes nn.Linear only; "
                        "LoRALinear and Conv2d layers stay FP32."
                    ),
                }
                m = result["precisions"]["int8_dynamic"]
                print(f"    RMSE={m_int8['rmse']:.4f}  AbsRel={m_int8['abs_rel']:.4f}  "
                      f"delta1={m_int8['delta1']:.4f}  "
                      f"mem={m['memory_mb']:.1f}MB  "
                      f"ms/img={m_int8['inference_ms_per_image']:.2f}")
                del model_int8
            except Exception as exc:
                print(f"  [warn] INT8 PTQ failed: {exc}")
                result["precisions"]["int8_dynamic"] = {"error": str(exc)}

        del model
        all_results.append(result)

    # ── print comparison table and save ──────────────────────────────────────
    print_summary(all_results)
    save_results(all_results, out_dir)
    print(f"\nAll done. Results: {out_dir}/")


if __name__ == "__main__":
    main()
