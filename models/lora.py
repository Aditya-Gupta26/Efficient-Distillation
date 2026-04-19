"""
Minimal LoRA implementation for Swin-Tiny attention layers.

LoRA (Low-Rank Adaptation) adds a pair of low-rank matrices A and B to a
frozen linear layer:

    W' x = Wx + (alpha/r) * B(Ax)

where:
    W  — frozen original weight  (out, in)
    A  — trainable  (r, in)       init: normal(0, 0.01)
    B  — trainable  (out, r)      init: zeros  → delta starts at 0
    r  — rank (4–16 typical)
    alpha — scaling factor

Only A and B are updated during training.

Target modules in Swin-Tiny:
    "qkv"  — fused Q/K/V projection inside each WindowAttention block
              (nn.Linear, in=dim, out=3*dim)

Usage:
    apply_lora(model.student, r=4, alpha=1.0, target_modules=["qkv"])
    params = get_lora_params(model.student)   # for optimizer
    n = count_lora_params(model.student)      # for logging
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with frozen base weights + LoRA delta.

    Args:
        linear: original (frozen) nn.Linear to wrap.
        r:      LoRA rank.
        alpha:  LoRA scaling factor (effective scale = alpha / r).
    """

    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.r     = r
        self.scale = alpha / r

        in_features  = linear.in_features
        out_features = linear.out_features

        # Store frozen base weights as non-parameter buffers so they are
        # moved with .to(device) but not returned by .parameters().
        self.register_buffer("weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Init A with small normal; B stays zero so delta starts at 0
        nn.init.normal_(self.lora_A, std=1.0 / math.sqrt(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base  = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + delta

    def extra_repr(self) -> str:
        out, inp = self.weight.shape
        return f"in={inp}, out={out}, r={self.r}, scale={self.scale:.4f}"


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def apply_lora(
    model:          nn.Module,
    r:              int         = 4,
    alpha:          float       = 1.0,
    target_modules: List[str]   = ("qkv",),
) -> None:
    """
    Replace every nn.Linear whose attribute name ends with one of
    *target_modules* with a LoRALinear wrapper, in-place.

    The base weights are frozen (stored as buffers); only LoRA A/B are
    added to the trainable parameter set.

    Args:
        model:          Module to patch (e.g. model.student).
        r:              LoRA rank.
        alpha:          LoRA scaling factor.
        target_modules: list of attribute name suffixes to target.
                        For Swin-Tiny use ["qkv"].
    """
    replaced = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        attr = full_name.split(".")[-1]
        if attr not in target_modules:
            continue

        # Navigate to parent and replace the attribute
        parts  = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        lora_layer = LoRALinear(module, r=r, alpha=alpha)
        setattr(parent, parts[-1], lora_layer)
        replaced += 1

    if replaced == 0:
        import warnings
        warnings.warn(
            f"apply_lora: no layers matched target_modules={target_modules}. "
            "Check that target_modules matches the actual attribute names in your model."
        )
    else:
        print(f"[lora] Replaced {replaced} Linear layers with LoRALinear (r={r}, alpha={alpha}).")


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return only LoRA A/B parameters from all LoRALinear layers."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def count_lora_params(model: nn.Module) -> int:
    """Total number of trainable LoRA parameters in the model."""
    return sum(p.numel() for p in get_lora_params(model))
