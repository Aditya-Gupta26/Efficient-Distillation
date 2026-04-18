"""
Student Model: Swin-Tiny (~28M params) backbone.

Swin-Tiny output channels per stage:
    Stage 0  (1/4  resolution) : 96
    Stage 1  (1/8  resolution) : 192
    Stage 2  (1/16 resolution) : 384
    Stage 3  (1/32 resolution) : 768

Uses the same single-model pattern as SwinTeacher:
  - forward_intermediates() for per-stage feature maps (NCHW)
  - self.model(x)           for logits through the intact pretrained head
    (which includes LayerNorm + AdaptiveAvgPool + Linear, not just a bare Linear)
"""

import torch
import torch.nn as nn
from timm import create_model


STUDENT_STAGE_CHANNELS = [96, 192, 384, 768]

SWIN_TINY_MODEL = "swin_tiny_patch4_window7_224"


class SwinStudentTiny(nn.Module):
    """
    Swin-Tiny student that mirrors the interface of :class:`SwinTeacher`:
    returns per-stage feature maps + optional classification logits.

    Args:
        pretrained (bool): Load ImageNet-1K pretrained weights via timm.
            Useful as a warm-start before distillation.
        num_classes (int): Number of output classes (default: 1000).
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stage_channels = STUDENT_STAGE_CHANNELS

        # Single model — intact pretrained head (LayerNorm + pool + Linear).
        # Do NOT pass num_classes here so the pretrained head is preserved.
        self.model = create_model(SWIN_TINY_MODEL, pretrained=pretrained)

        # Optionally replace the final classifier if num_classes != 1000
        if num_classes != 1000:
            in_features = self.model.head.fc.in_features  # 768 for Swin-Tiny
            self.model.reset_classifier(num_classes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Image tensor (B, 3, H, W).

        Returns:
            features (list[Tensor]): Per-stage feature maps (B, C_i, H_i, W_i).
            logits   (Tensor | None): Classification logits (B, num_classes), or None.
        """
        # forward_intermediates returns (final_features, intermediates_list)
        _, features = self.model.forward_intermediates(
            x,
            indices=[0, 1, 2, 3],   # all 4 Swin stages
            output_fmt="NCHW",       # (B, C, H, W) — no permute needed
        )

        logits = None
        if self.num_classes > 0:
            logits = self.model(x)   # full forward through intact pretrained head

        return features, logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
