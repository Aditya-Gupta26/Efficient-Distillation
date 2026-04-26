"""
StudentWithDPT: distilled Swin-Tiny backbone (frozen) + DPT depth head.

Loading order:
  1. Instantiate SwinStudentTiny with num_classes=0 (no classification head).
  2. Load student weights from a distillation checkpoint
     (expected key: "student", written by utils/checkpoint.py).
  3. Freeze all student parameters (controlled by freeze_student flag).
  4. Attach DPTDepthHead (trainable by default).

Forward:
  image (B,3,H,W) → student features → DPT neck+head → depth (B,1,H,W)
  The depth map is bilinearly upsampled to match the input spatial size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.student import SwinStudentTiny
from models.dpt_head import DPTDepthHead


def _load_student_from_checkpoint(student: SwinStudentTiny, path: str) -> None:
    """Load student weights from a distillation training checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    # Checkpoint written by utils/checkpoint.py has key "student"
    if "student" in ckpt:
        state = ckpt["student"]
    else:
        # Bare state dict (e.g. exported manually)
        state = ckpt

    # Strip torch.compile prefix if checkpoint was saved from a compiled model
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing:
        # Head weights absent because num_classes=0 — expected, not a bug
        head_keys   = [k for k in missing if "head" in k]
        other_keys  = [k for k in missing if "head" not in k]
        if other_keys:
            print(f"[depth_model] WARNING: {len(other_keys)} unexpected missing keys: {other_keys[:5]}")
        if head_keys:
            print(f"[depth_model] Skipped {len(head_keys)} classification-head keys (num_classes=0, expected).")
    if unexpected:
        non_head = [k for k in unexpected if "head" not in k]
        head_keys = [k for k in unexpected if "head" in k]
        if head_keys:
            print(f"[depth_model] Skipped {len(head_keys)} classification-head keys in checkpoint (num_classes=0, expected).")
        if non_head:
            print(f"[depth_model] WARNING: {len(non_head)} truly unexpected keys in checkpoint: {non_head[:5]}")


class StudentWithDPT(nn.Module):
    """
    Frozen distilled student backbone + trainable DPT depth head.

    Args:
        student_checkpoint: path to distillation .pth file produced by
                            the imagenet distillation training.
        dpt_pretrained:     if True, load DPT neck+head from HuggingFace.
        freeze_student:     if True, freeze all student parameters.
    """

    def __init__(
        self,
        student_checkpoint: str,
        dpt_pretrained:     bool = True,
        freeze_student:     bool = True,
    ):
        super().__init__()

        # Student backbone (no classification head needed)
        self.student = SwinStudentTiny(pretrained=False, num_classes=0)
        _load_student_from_checkpoint(self.student, student_checkpoint)

        self.freeze_student = freeze_student
        if freeze_student:
            for p in self.student.parameters():
                p.requires_grad = False
            self.student.eval()

        # DPT depth head — trainable
        self.dpt_head = DPTDepthHead(pretrained=dpt_pretrained)

        # Per-stage layer norms (commented out — may cause scale explosions with
        # some student checkpoints; re-enable if DPT neck diverges early).
        # self.feat_norms = nn.ModuleList([
        #     nn.LayerNorm(c, elementwise_affine=False) for c in [96, 192, 384, 768]
        # ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised input images.

        Returns:
            depth: (B, 1, H, W) predicted depth map, upsampled to input size.
        """
        # Student backbone produces 4 NCHW feature maps
        features, _ = self.student(x)

        # features = [
        #     norm(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #     for f, norm in zip(features, self.feat_norms)
        # ]

        # DPT neck+head → coarse depth map
        depth = self.dpt_head(features)

        # Upsample to match input resolution
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(
                depth,
                size=x.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )

        return depth

    def train(self, mode: bool = True):
        super().train(mode)
        # If student is frozen, keep it in eval regardless of overall mode
        # (frozen BN/dropout should not be in train mode)
        if self.freeze_student:
            self.student.eval()
        return self
