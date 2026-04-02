"""
Distillation Losses.

Supports three complementary loss terms:

1. **Feature-level loss** (MSE / L2 between adapted student and teacher maps)
   – Applied at each active stage after the adapter projects student features
     into teacher channel space.

2. **Attention Transfer loss** (AT)
   – Matches the spatial attention maps (sum of squared activations across
     channels) between teacher and student, encouraging the student to "look"
     at the same regions as the teacher.

3. **Logit-level KD loss** (KL-divergence with temperature)
   – Classic Hinton et al. knowledge distillation on softened class
     probabilities.

The total loss is a weighted sum:

    L = w_feat * L_feat + w_at * L_at + w_kd * L_kd + w_task * L_task
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DistillationLoss(nn.Module):
    """
    Combined knowledge distillation loss.

    Args:
        w_feat  (float): Weight for feature-level MSE loss.
        w_at    (float): Weight for attention-transfer loss.
        w_kd    (float): Weight for logit-level KD loss.
        w_task  (float): Weight for task-specific (e.g., detection) loss.
        temperature (float): Softmax temperature for KD loss (τ).
        feat_loss_type (str): ``"mse"`` or ``"cosine"`` for feature matching.
    """

    def __init__(
        self,
        w_feat: float = 1.0,
        w_at: float = 0.5,
        w_kd: float = 1.0,
        w_task: float = 1.0,
        temperature: float = 4.0,
        feat_loss_type: str = "mse",
    ):
        super().__init__()
        self.w_feat = w_feat
        self.w_at = w_at
        self.w_kd = w_kd
        self.w_task = w_task
        self.temperature = temperature
        assert feat_loss_type in ("mse", "cosine"), (
            "feat_loss_type must be 'mse' or 'cosine'."
        )
        self.feat_loss_type = feat_loss_type

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    def feature_loss(
        self,
        adapted_student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """MSE or cosine distance between adapted-student and teacher feature maps."""
        total = torch.tensor(0.0, device=teacher_feats[0].device)
        for s_feat, t_feat in zip(adapted_student_feats, teacher_feats):
            if self.feat_loss_type == "mse":
                total = total + F.mse_loss(s_feat, t_feat.detach())
            else:
                # Cosine loss: flatten spatial dims
                B, C, H, W = s_feat.shape
                s_flat = s_feat.view(B, C, -1).permute(0, 2, 1)   # (B, HW, C)
                t_flat = t_feat.detach().view(B, C, -1).permute(0, 2, 1)
                total = total + (1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1)).mean()
        return total / max(len(adapted_student_feats), 1)

    def attention_transfer_loss(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Attention Transfer (Zagoruyko & Komodakis, 2017).
        Matches normalised attention maps: A(F) = ||F||_2 summed over channels.
        """
        total = torch.tensor(0.0, device=teacher_feats[0].device)
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            s_att = self._attention_map(s_feat)
            t_att = self._attention_map(t_feat.detach())
            total = total + F.mse_loss(s_att, t_att)
        return total / max(len(student_feats), 1)

    @staticmethod
    def _attention_map(feat: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H*W) normalised attention."""
        B = feat.shape[0]
        att = feat.pow(2).mean(dim=1)       # (B, H, W)
        att = att.view(B, -1)               # (B, H*W)
        att = F.normalize(att, p=2, dim=1)  # L2-normalise
        return att

    def kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL-divergence KD loss with temperature scaling."""
        τ = self.temperature
        s_log_prob = F.log_softmax(student_logits / τ, dim=-1)
        t_prob     = F.softmax(teacher_logits.detach() / τ, dim=-1)
        return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (τ ** 2)

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------
    def forward(
        self,
        adapted_student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
        student_feats_raw: List[torch.Tensor],
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        task_loss: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute the combined distillation loss.

        Args:
            adapted_student_feats: Student features projected by adapters
                into teacher channel space.  List[(B, C_t, H, W)].
            teacher_feats: Raw teacher feature maps. List[(B, C_t, H, W)].
            student_feats_raw: Original (un-adapted) student feature maps,
                used for the attention transfer loss.
            student_logits: (B, num_classes) – optional.
            teacher_logits: (B, num_classes) – optional.
            task_loss: Scalar task-specific loss (e.g., from detection head).

        Returns:
            dict with keys:
                ``"total"``, ``"feat"``, ``"at"``, ``"kd"``, ``"task"``
        """
        losses = {}

        # Feature-level distillation
        l_feat = self.feature_loss(adapted_student_feats, teacher_feats)
        losses["feat"] = l_feat

        # Attention transfer
        l_at = self.attention_transfer_loss(student_feats_raw, teacher_feats)
        losses["at"] = l_at

        # Logit-level KD
        l_kd = torch.tensor(0.0, device=l_feat.device)
        if student_logits is not None and teacher_logits is not None:
            l_kd = self.kd_loss(student_logits, teacher_logits)
        losses["kd"] = l_kd

        # Task-specific loss (e.g., detection / segmentation)
        l_task = task_loss if task_loss is not None else torch.tensor(0.0, device=l_feat.device)
        losses["task"] = l_task

        # Weighted total
        total = (
            self.w_feat * l_feat
            + self.w_at  * l_at
            + self.w_kd  * l_kd
            + self.w_task * l_task
        )
        losses["total"] = total
        return losses
