"""
Feature Adapters for Knowledge Distillation.

Adapters bridge the dimensional mismatch between the teacher's richer feature
maps and the student's narrower ones.  Each adapter is a lightweight
convolutional bottleneck that:

    1. Projects the *student* features up to the teacher's channel width.
    2. (Optionally) applies spatial alignment if the feature-map resolutions
       differ (e.g., due to different stride schedules).

Architecture per adapter:
    Conv2d(C_s → C_s)  BN  ReLU
    Conv2d(C_s → C_t)  BN

This keeps the adapter cheap (~2 conv layers per stage) while still allowing
the distillation signal to be expressed in the teacher's embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SingleStageAdapter(nn.Module):
    """
    A per-stage adapter that projects student features into teacher space.

    Args:
        student_channels (int): Channel width of the student at this stage.
        teacher_channels (int): Channel width of the teacher at this stage.
        use_spatial_align (bool): If True, the adapter accepts an optional
            ``target_size`` argument in forward() and bilinearly resizes the
            projected feature map to match teacher spatial dims.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        use_spatial_align: bool = True,
    ):
        super().__init__()
        self.use_spatial_align = use_spatial_align

        mid_channels = max(student_channels, student_channels // 2)

        self.proj = nn.Sequential(
            # Intra-channel mixing (depth-wise separable style)
            nn.Conv2d(student_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # Channel expansion to teacher width
            nn.Conv2d(mid_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        student_feat: torch.Tensor,
        target_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_feat: (B, C_s, H, W)
            target_size:  (H_t, W_t) — if provided and use_spatial_align=True,
                          the output is resized to this spatial resolution.

        Returns:
            Tensor of shape (B, C_t, H_t, W_t) or (B, C_t, H, W).
        """
        out = self.proj(student_feat)
        if self.use_spatial_align and target_size is not None:
            if out.shape[-2:] != target_size:
                out = F.interpolate(
                    out, size=target_size, mode="bilinear", align_corners=False
                )
        return out


class FeatureAdapter(nn.Module):
    """
    Container that holds one :class:`SingleStageAdapter` per distillation
    stage.  Call ``forward(student_features, teacher_features)`` to obtain
    a list of adapted student tensors that are aligned with teacher tensors.

    Args:
        student_channels (List[int]): Per-stage channel widths of the student.
        teacher_channels (List[int]): Per-stage channel widths of the teacher.
        stages (List[int] | None): Which stages to attach adapters to.
            Defaults to all stages.
        use_spatial_align (bool): Passed to each :class:`SingleStageAdapter`.
    """

    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        stages: Optional[List[int]] = None,
        use_spatial_align: bool = True,
    ):
        super().__init__()
        assert len(student_channels) == len(teacher_channels), (
            "student_channels and teacher_channels must have the same length."
        )
        self.num_stages = len(student_channels)
        self.active_stages = stages if stages is not None else list(range(self.num_stages))

        adapters = {}
        for i in self.active_stages:
            adapters[f"stage_{i}"] = SingleStageAdapter(
                student_channels=student_channels[i],
                teacher_channels=teacher_channels[i],
                use_spatial_align=use_spatial_align,
            )
        self.adapters = nn.ModuleDict(adapters)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Project each active student stage into teacher feature space.

        Args:
            student_features: List of (B, C_s_i, H_i, W_i) tensors.
            teacher_features: List of (B, C_t_i, H_i, W_i) tensors.
                              Used only for spatial dimension reference.

        Returns:
            adapted (List[Tensor]): Projected student features, one per
                active stage.  Non-active stages are returned as-is.
        """
        adapted = list(student_features)  # shallow copy
        for i in self.active_stages:
            t_feat = teacher_features[i]
            target_size = t_feat.shape[-2:]
            adapted[i] = self.adapters[f"stage_{i}"](
                student_features[i], target_size=target_size
            )
        return adapted

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
