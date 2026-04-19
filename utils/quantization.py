"""
Post-Training Quantization (PTQ) utilities for the depth pipeline.

Supported methods
-----------------
ptq_dynamic
    Quantize Linear layers at inference time (no calibration needed).
    Fast to apply; ~2x memory reduction; slight accuracy drop.
    torch.quantization.quantize_dynamic — CPU-friendly.

ptq_static
    Quantize with a calibration pass on real data (better accuracy than dynamic).
    Requires a DataLoader for calibration.
    Works on CPU; GPU static PTQ needs torch.ao or x86 backend.

bnb_int8
    8-bit Linear via bitsandbytes (LLM.int8).
    Requires CUDA + bitsandbytes package.  Near-lossless on large models.

bnb_nf4
    4-bit NormalFloat via bitsandbytes (QLoRA style).
    Highest compression; recommended for inference-only experiments.

Usage
-----
    from utils.quantization import quantize_model

    # Evaluate student backbone in dynamic int8
    q_model = quantize_model(student, method="ptq_dynamic")

    # Evaluate full StudentWithDPT in dynamic int8
    q_model = quantize_model(depth_model, method="ptq_dynamic")

    # Static PTQ (needs calibration data)
    q_model = quantize_model(depth_model, method="ptq_static",
                             calibration_loader=val_loader, device=device)

Config flag (in YAML):
    quantization:
      enabled: true
      method:  "ptq_dynamic"   # ptq_dynamic | ptq_static | bnb_int8 | bnb_nf4
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def quantize_model(
    model:               nn.Module,
    method:              str = "ptq_dynamic",
    calibration_loader:  Optional[DataLoader] = None,
    device:              Optional[torch.device] = None,
    dtype:               str = "int8",
) -> nn.Module:
    """
    Quantize *model* in-place or return a new quantized copy.

    Args:
        model:              Module to quantize (should be in eval mode).
        method:             One of "ptq_dynamic", "ptq_static", "bnb_int8", "bnb_nf4".
        calibration_loader: DataLoader for static PTQ calibration (required for ptq_static).
        device:             Target device (used for static PTQ).
        dtype:              "int8" or "int4" (only used where method supports it).

    Returns:
        Quantized model (may be the same object mutated in-place, or a new one).
    """
    model.eval()

    if method == "ptq_dynamic":
        return _ptq_dynamic(model)
    elif method == "ptq_static":
        if calibration_loader is None:
            raise ValueError("ptq_static requires calibration_loader.")
        return _ptq_static(model, calibration_loader, device or torch.device("cpu"))
    elif method == "bnb_int8":
        return _bnb_quantize(model, bits=8)
    elif method == "bnb_nf4":
        return _bnb_quantize(model, bits=4)
    else:
        raise ValueError(
            f"Unknown quantization method: {method!r}. "
            "Choose from: ptq_dynamic, ptq_static, bnb_int8, bnb_nf4"
        )


# ------------------------------------------------------------------
# PTQ Dynamic
# ------------------------------------------------------------------

def _ptq_dynamic(model: nn.Module) -> nn.Module:
    """
    Dynamic quantization: convert nn.Linear weights to int8 at load time,
    activations quantized on-the-fly at inference.  No calibration needed.
    """
    q_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec = {nn.Linear},
        dtype        = torch.qint8,
        inplace      = False,
    )
    print("[quantization] Applied dynamic int8 PTQ to all nn.Linear layers.")
    return q_model


# ------------------------------------------------------------------
# PTQ Static
# ------------------------------------------------------------------

def _ptq_static(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    n_batches: int = 10,
) -> nn.Module:
    """
    Static PTQ: insert observer hooks, run calibration pass, then convert.

    Note: torch static PTQ runs on CPU.  Move model to CPU for calibration
    and back to device afterward if needed.
    """
    import copy

    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    # Use fbgemm backend (x86 CPU)
    model_cpu.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    torch.quantization.prepare(model_cpu, inplace=True)

    print(f"[quantization] Running calibration ({n_batches} batches)...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            images = batch["images"].cpu()
            model_cpu(images)

    torch.quantization.convert(model_cpu, inplace=True)
    print("[quantization] Static int8 PTQ complete.")
    return model_cpu


# ------------------------------------------------------------------
# bitsandbytes (int8 / NF4)
# ------------------------------------------------------------------

def _bnb_quantize(model: nn.Module, bits: int = 8) -> nn.Module:
    """
    Replace nn.Linear layers with bitsandbytes quantized equivalents.
    Requires: pip install bitsandbytes   and CUDA.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for bnb_int8 / bnb_nf4 quantization.\n"
            "Install with: pip install bitsandbytes"
        )

    if bits == 8:
        linear_cls = bnb.nn.Linear8bitLt
        kwargs     = {"has_fp16_weights": False}
        tag        = "int8"
    elif bits == 4:
        linear_cls = bnb.nn.Linear4bit
        kwargs     = {"quant_type": "nf4", "compute_dtype": torch.float16}
        tag        = "nf4"
    else:
        raise ValueError(f"bits must be 8 or 4, got {bits}")

    replaced = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        parts  = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        q_layer = linear_cls(
            module.in_features,
            module.out_features,
            bias = module.bias is not None,
            **kwargs,
        )
        # Copy weights (bnb will quantize on first forward)
        q_layer.weight = module.weight
        if module.bias is not None:
            q_layer.bias = module.bias

        setattr(parent, parts[-1], q_layer)
        replaced += 1

    print(f"[quantization] Replaced {replaced} Linear layers with bitsandbytes {tag}.")
    return model


# ------------------------------------------------------------------
# Convenience: compare model size before/after
# ------------------------------------------------------------------

def model_size_mb(model: nn.Module) -> float:
    """Approximate model size in megabytes (parameter memory only)."""
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)
