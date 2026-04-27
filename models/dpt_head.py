"""
DPT depth head for the distilled Swin-Tiny student backbone.

Architecture: Intel/dpt-swinv2-tiny-256 neck + head (randomly initialised).

Why random initialisation (pretrained=False):
  Intel's pretrained weights were fitted to SwinV2-Tiny features.
  Our student is a distilled Swin-Tiny v1 with different feature statistics.
  Random (Xavier) weights plus the InstanceNorm input normalisation below
  start the neck in a well-conditioned regime from the first training step.

Why align_corners=False patch:
  The HuggingFace DPTFeatureFusionLayer uses
      F.interpolate(..., scale_factor=2, align_corners=True)
  in every fusion step.  align_corners=True is a known source of NaN on Apple
  MPS (PyTorch / Metal backend limitation).  After the neck is constructed we
  set align_corners=False on every fusion layer — this is the only change
  needed to make training stable on MPS, CUDA, and CPU alike.

Neck architecture (DPTNeck for dpt-swinv2-tiny-256):
  convs           : 4 × Conv2d(C_i → 256, 3×3)  where C = [96,192,384,768]
  fusion_stage    : 4 × DPTFeatureFusionLayer (residual Conv blocks + ×2 upsample)

Head architecture (DPTDepthEstimationHead):
  Conv2d(256→128, 3×3) → Upsample(×2) → Conv2d(128→32, 3×3) → Conv2d(32→1, 1×1)
  Internal ReLUs replaced with Identity; F.softplus is the sole output activation.

Input  : list of 4 NCHW tensors  [(B,96,56,56), (B,192,28,28),
                                   (B,384,14,14), (B,768,7,7)]
Output : (B, 1, H_out, W_out)  positive depth in metres.
         Caller (StudentWithDPT) upsamples to the original image size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "Intel/dpt-swinv2-tiny-256"

# Swin-Tiny stage output channels — match DPT neck Conv input sizes exactly
STUDENT_CHANNELS = [96, 192, 384, 768]


class DPTDepthHead(nn.Module):
    """
    DPT neck + head with three targeted fixes for our setup:

      Fix 1 — InstanceNorm2d (affine) before the neck
               Normalises each student feature channel to mean=0, std=1 per
               sample so the randomly-initialised neck convolutions start in a
               well-conditioned range regardless of backbone output scale.

      Fix 2 — align_corners=False in every DPTFeatureFusionLayer
               Prevents NaN on Apple MPS where align_corners=True triggers a
               Metal backend bug in F.interpolate.

      Fix 3 — ReLU → Identity inside head.head Sequential + F.softplus output
               Prevents the dead-gradient / constant-output failure that occurs
               when the final Conv(32→1) output is uniformly negative for
               out-of-distribution student features.

    Args:
        pretrained: load HuggingFace weights (default False — use random init).
                    Only set True if the upstream backbone is SwinV2-Tiny.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        from transformers import DPTForDepthEstimation, DPTConfig
        if pretrained:
            full = DPTForDepthEstimation.from_pretrained(MODEL_ID)
        else:
            cfg  = DPTConfig.from_pretrained(MODEL_ID)
            full = DPTForDepthEstimation(cfg)   # random Xavier weights

        self.neck        = full.neck
        self.head        = full.head
        self._dpt_config = full.config
        del full

        # ── Fix 1: per-stage input normalisation ─────────────────────────────
        # InstanceNorm2d normalises each (sample, channel) pair over H×W.
        # affine=True adds a learned per-channel scale + shift; these parameters
        # are trained together with the rest of the DPT head.
        self.feature_norms = nn.ModuleList([
            nn.InstanceNorm2d(c, affine=True) for c in STUDENT_CHANNELS
        ])

        # ── Fix 2: replace align_corners=True → False in all fusion layers ───
        # DPTFeatureFusionLayer stores self.align_corners and passes it to
        # F.interpolate.  Flipping the attribute is the minimal targeted fix.
        for m in self.neck.modules():
            if hasattr(m, "align_corners"):
                m.align_corners = False

        # ── Fix 3: remove dead-ReLU activations from the head Sequential ─────
        # head.head layout after this patch:
        #   [0] Conv2d(256, 128, 3×3)
        #   [1] Upsample(×2, bilinear)
        #   [2] Conv2d(128,  32, 3×3)
        #   [3] Identity()    ← was ReLU
        #   [4] Conv2d( 32,   1, 1×1)
        #   [5] Identity()    ← was ReLU  (softplus is applied in forward())
        for idx in range(len(self.head.head)):
            if isinstance(self.head.head[idx], nn.ReLU):
                self.head.head[idx] = nn.Identity()

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 4 NCHW tensors from the student backbone's
                      forward_intermediates(indices=[0,1,2,3], output_fmt='NCHW').

        Returns:
            depth: (B, 1, H_out, W_out) — positive depth values.
                   StudentWithDPT will bilinearly upsample to input resolution.
        """
        # Fix 1: normalise student features before the pretrained neck
        features = [self.feature_norms[i](f) for i, f in enumerate(features)]

        # DPT neck: channel projection (convs) → feature fusion (fusion_stage)
        fused    = self.neck(tuple(features))

        # DPT head: select finest fused feature, run Conv stack
        # head_in_index = -1 for this config (finest = last element)
        selected = fused[self._dpt_config.head_in_index]
        depth    = self.head.head(selected)   # (B, 1, H_out, W_out)

        # Fix 3: smooth positive activation — gradient always non-zero
        depth = F.softplus(depth)

        return depth
