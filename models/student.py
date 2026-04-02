"""
Student Model: Swin-Tiny (~28M params) backbone.

Swin-Tiny output channels per stage:
    Stage 0  (1/4  resolution) : 96
    Stage 1  (1/8  resolution) : 192
    Stage 2  (1/16 resolution) : 384
    Stage 3  (1/32 resolution) : 768
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
        num_classes (int): Number of output classes.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 80,
    ):
        super().__init__()
        self.stage_channels = STUDENT_STAGE_CHANNELS

        self.backbone = create_model(
            SWIN_TINY_MODEL,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        in_features = self.stage_channels[-1]
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.head = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Image tensor (B, 3, H, W).

        Returns:
            features (list[Tensor]): Per-stage feature maps.
            logits   (Tensor | None): Classification logits, or None.
        """
        features = self.backbone(x)

        # timm Swin features_only uses channels-last (B, H, W, C).
        # Permute to standard (B, C, H, W) for compatibility with adapters / losses.
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        logits = None
        if self.head is not None:
            logits = self.head(features[-1])

        return features, logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
