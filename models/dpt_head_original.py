"""
DPT depth head extracted from Intel/dpt-swinv2-tiny-256.

Takes the 4-stage NCHW feature maps from the Swin-Tiny student backbone
(channels [96, 192, 384, 768]) and produces a dense depth map.

The backbone is discarded; only the DPT neck (reassemble + fusion) and
the depth estimation head are kept and loaded with pretrained weights.

Input  : list of 4 tensors [(B,96,H0,W0), (B,192,H1,W1), (B,384,H2,W2), (B,768,H3,W3)]
Output : (B, 1, H_in, W_in) depth map in whatever units the head produces
         (caller is responsible for final bilinear upsample to input resolution)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "Intel/dpt-swinv2-tiny-256"

# Student Swin-Tiny stage channels — must match DPT neck_hidden_sizes
STUDENT_CHANNELS = [96, 192, 384, 768]


class DPTDepthHead(nn.Module):
    """
    Wraps the pretrained DPT neck + head from Intel/dpt-swinv2-tiny-256.

    The neck contains:
      - reassemble_stage : per-stage channel projection + spatial resize
      - channel_projection: maps each stage to fusion_hidden_size (256)
      - fusion_stage      : progressive feature fusion

    The head is a lightweight Conv stack that outputs a single-channel
    depth map from the finest fused feature.

    Args:
        pretrained: if True, load weights from HuggingFace hub.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            from transformers import DPTForDepthEstimation
            full = DPTForDepthEstimation.from_pretrained(MODEL_ID)
        else:
            from transformers import DPTForDepthEstimation, DPTConfig
            cfg = DPTConfig.from_pretrained(MODEL_ID)
            full = DPTForDepthEstimation(cfg)

        # Extract neck and prediction head; discard backbone
        self.neck = full.neck
        self.head = full.head

        # Remember the config so we can query head_in_index if needed
        self._dpt_config = full.config

        del full

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 4 NCHW tensors from student's forward_intermediates()
                      [stage0 (96ch), stage1 (192ch), stage2 (384ch), stage3 (768ch)]

        Returns:
            depth: (B, 1, H_neck, W_neck) — not yet upsampled to input resolution.
                   Caller should F.interpolate to the original image size.
        """
        # DPT neck expects a tuple/list of feature maps.
        # For Swin-based DPT the reassemble layers work on NCHW directly.
        hidden_states = tuple(features)

        # neck: reassemble → channel_proj → fusion  →  list of fused tensors
        fused = self.neck(hidden_states)

        # head: picks fused[head_in_index] and runs Conv stack
        depth = self.head(fused)

        return depth