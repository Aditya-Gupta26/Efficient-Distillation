"""
Intel/dpt-swinv2-tiny-256 — Zero-shot baseline evaluation on NYU Depth V2.

Downloads the complete pretrained Intel DPT model from HuggingFace and
evaluates it on the NYU Depth V2 validation set.  No training occurs.

Purpose: establish a reference score (the "ceiling" for a model trained on
real depth data) so we can compare our distillation-based approaches against
a production-quality baseline.

Model  : Intel/dpt-swinv2-tiny-256  (SwinV2-Tiny backbone + DPT head)
Data   : NYU Depth V2 val split (144 images, 224×224 for loading,
         resized to 256×256 as required by the Intel model)
Metrics: RMSE, AbsRel, log-RMSE, SILog, δ1, δ2, δ3

Usage:
    python customization/Intel_dpt_swin_tiny_256.py
    python customization/Intel_dpt_swin_tiny_256.py --nyu_root data/nyu_depth_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import compute_depth_metrics, get_fixed_val_indices, FIXED_SEED, N_EVAL_IMAGES
from data.nyu_depth_dataset import NyuDepthDataset


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID      = "Intel/dpt-swinv2-tiny-256"
INTEL_SIZE    = 256    # The Intel model was trained on 256×256 images
# Local cache dir — model is downloaded once then loaded from here on all
# subsequent runs, avoiding repeated HuggingFace API calls and rate-limit errors.
HF_CACHE_DIR  = Path("checkpoints/hf_models/intel_dpt_swinv2_tiny_256")


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation (same stats used by the Intel model's processor)
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalise_to_pil(tensor: torch.Tensor):
    """Convert a normalised (3,H,W) tensor to a PIL Image for the HF processor."""
    from PIL import Image
    img = (tensor.cpu() * _STD + _MEAN).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_intel_model(device: torch.device):
    """
    Load the pretrained Intel DPT model.

    On the FIRST run: downloads from HuggingFace and saves to HF_CACHE_DIR so
    that future runs never hit the network again (avoids unauthenticated-request
    rate-limit errors from the HF Hub).

    On SUBSEQUENT runs: loads entirely from the local cache with
    local_files_only=True — no network request is made at all.

    The model has its own SwinV2-Tiny backbone (NOT our distilled student).
    We use the HuggingFace DPTImageProcessor for correct preprocessing so
    the model receives images in exactly the format it was trained on.
    """
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    cache_dir = HF_CACHE_DIR.resolve()

    if cache_dir.exists() and any(cache_dir.iterdir()):
        # ── Load from local cache (fast, no network) ──────────────────────────
        print(f"Loading Intel DPT from local cache: {cache_dir}")
        processor = DPTImageProcessor.from_pretrained(
            str(cache_dir), local_files_only=True)
        model     = DPTForDepthEstimation.from_pretrained(
            str(cache_dir), local_files_only=True)
    else:
        # ── First run: download from HuggingFace then save locally ────────────
        print(f"Downloading {MODEL_ID} from HuggingFace (one-time) …")
        processor = DPTImageProcessor.from_pretrained(MODEL_ID)
        model     = DPTForDepthEstimation.from_pretrained(MODEL_ID)
        cache_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(str(cache_dir))
        model.save_pretrained(str(cache_dir))
        print(f"Model cached to {cache_dir}  (future runs load from here)")

    model.to(device).eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Intel DPT loaded — {total:,} parameters\n")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_depth(
    model,
    processor,
    pil_image,
    target_size: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """
    Run the Intel DPT model on a single PIL image.

    The processor resizes + normalises to 256×256 internally.
    We then resize the output back to target_size (224×224) so it can be
    compared fairly against ground-truth depth maps from NyuDepthDataset.

    Returns:
        depth_np: (H, W) numpy array in model-relative units.
                  We normalise to metre scale by min-max for visual comparison
                  but use the raw output for metric computation relative to GT.
    """
    # processor handles resizing, normalisation, and tensor conversion
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs       = model(**inputs)
    pred_depth    = outputs.predicted_depth          # (1, H_out, W_out)

    # Resize prediction to match our ground-truth resolution
    pred_resized  = F.interpolate(
        pred_depth.unsqueeze(1),
        size=target_size,
        mode="bicubic",
        align_corners=False,
    ).squeeze()   # (H, W)

    return pred_resized.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Intel/dpt-swinv2-tiny-256 on NYU Depth V2 val")
    p.add_argument("--nyu_root", default="data/nyu_depth_v2")
    p.add_argument("--img_size", type=int, default=224, help="Spatial size for GT depth maps")
    p.add_argument("--results",  default="checkpoints/customization/intel_dpt_results.json",
                   help="Where to save metric results (JSON)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_intel_model(device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    # We use the full val split for metrics, and the fixed 10 images for visuals.
    dataset = NyuDepthDataset(root=args.nyu_root, split="val", img_size=args.img_size)
    print(f"NYU val split: {len(dataset)} images\n")

    # ── Full evaluation — all 144 val images ──────────────────────────────────
    print("Running inference on full val split …")

    metric_sums = {k: 0.0 for k in ["rmse", "abs_rel", "log_rmse", "silog",
                                      "delta1", "delta2", "delta3"]}
    n = 0

    for idx in range(len(dataset)):
        sample    = dataset[idx]
        pil_img   = denormalise_to_pil(sample["images"])
        gt_depth  = sample["depths"]                  # (H, W) tensor, metres

        pred = predict_depth(model, processor, pil_img,
                             target_size=(args.img_size, args.img_size), device=device)

        # The Intel model outputs relative depth — scale to match GT metre range
        # by aligning the median of valid pixels.
        gt_valid   = gt_depth[gt_depth > 0]
        pred_valid = pred[gt_depth > 0]
        if gt_valid.numel() > 0 and pred_valid.median() > 0:
            scale = gt_valid.median() / pred_valid.median()
            pred  = pred * scale

        m = compute_depth_metrics(pred, gt_depth)
        for k in metric_sums:
            metric_sums[k] += m[k]
        n += 1

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(dataset)}  running RMSE={metric_sums['rmse']/n:.4f}",
                  end="\r", flush=True)

    print()

    # ── Average metrics ───────────────────────────────────────────────────────
    avg = {k: v / n for k, v in metric_sums.items()}

    print("\n" + "=" * 55)
    print(f"  Intel/dpt-swinv2-tiny-256  —  NYU Depth V2 val")
    print("=" * 55)
    print(f"  RMSE         : {avg['rmse']:.4f} m")
    print(f"  AbsRel       : {avg['abs_rel']:.4f}")
    print(f"  log-RMSE     : {avg['log_rmse']:.4f}")
    print(f"  SILog        : {avg['silog']:.4f}")
    print(f"  δ1 (< 1.25)  : {avg['delta1']:.4f}")
    print(f"  δ2 (< 1.25²) : {avg['delta2']:.4f}")
    print(f"  δ3 (< 1.25³) : {avg['delta3']:.4f}")
    print("=" * 55)

    # ── Save results to JSON for validation.py ────────────────────────────────
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"model": MODEL_ID, "metrics": avg}, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
