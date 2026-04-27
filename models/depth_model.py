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


def _load_student_from_checkpoint(student: SwinStudentTiny, path: str) -> bool:
    """Load student weights from a distillation training checkpoint.

    Returns:
        True  — weights loaded successfully (no NaN).
        False — checkpoint has NaN weights (diverged training); caller should
                fall back to timm pretrained weights.

    Previously this raised RuntimeError on NaN.  It now warns and returns False
    so the caller can fall back gracefully — useful when we still want to run
    the pipeline for architecture verification or comparison purposes.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Checkpoint written by utils/checkpoint.py has key "student"
    state = ckpt["student"] if "student" in ckpt else ckpt

    # Strip torch.compile prefix if checkpoint was saved from a compiled model
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # NaN check — a corrupted checkpoint (diverged distillation) must be caught
    # before any weights are applied to the model.
    nan_keys = [k for k, v in state.items() if torch.is_tensor(v) and torch.isnan(v).any()]
    if nan_keys:
        raise RuntimeError(
            f"[depth_model] CORRUPT CHECKPOINT: {len(nan_keys)}/{len(state)} tensors "
            f"in '{path}' contain NaN. Distillation diverged at epoch "
            f"{ckpt.get('epoch', '?')}. Use best.pth from the distillation run."
        )

    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing:
        head_keys  = [k for k in missing if "head" in k]
        other_keys = [k for k in missing if "head" not in k]
        if other_keys:
            print(f"[depth_model] WARNING: {len(other_keys)} unexpected missing keys: {other_keys[:5]}", flush=True)
        if head_keys:
            print(f"[depth_model] Skipped {len(head_keys)} classification-head keys (num_classes=0, expected).", flush=True)
    if unexpected:
        print(f"[depth_model] WARNING: {len(unexpected)} unexpected keys in checkpoint.", flush=True)

    # Print a summary so the user can confirm which distilled checkpoint is loaded
    total_params = sum(p.numel() for p in student.parameters())
    print(f"[depth_model] Distilled student loaded from  : {path}", flush=True)
    print(f"[depth_model] Checkpoint epoch               : {ckpt.get('epoch', 'N/A')}", flush=True)
    print(f"[depth_model] Best distillation metric       : {ckpt.get('best_metric', 'N/A')}", flush=True)
    print(f"[depth_model] Student parameters             : {total_params:,}", flush=True)


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
        student_checkpoint: str | None,
        dpt_pretrained:     bool = True,
        freeze_student:     bool = True,
    ):
        super().__init__()

        # Student backbone (no classification head needed).
        # If student_checkpoint is None, fall back to the publicly-pretrained
        # timm Swin-Tiny weights — architecturally identical to the distilled
        # model and useful for pipeline testing when the distilled checkpoint
        # is unavailable or corrupted.
        # Always load the distilled student from the provided checkpoint.
        # The checkpoint path is required — no silent fallback to any other weights.
        self.student = SwinStudentTiny(pretrained=False, num_classes=0)
        _load_student_from_checkpoint(self.student, student_checkpoint)

        # ── Previous timm-pretrained fallback (kept for reference) ───────────
        # This was used when student_checkpoint=None or when NaN weights were
        # detected, to allow pipeline testing without a valid distilled checkpoint.
        # Now that we have a valid distilled student (best.pth), we always use it.
        #
        # if student_checkpoint is None:
        #     print("[depth_model] No student checkpoint — using timm pretrained Swin-Tiny.")
        #     self.student = SwinStudentTiny(pretrained=True, num_classes=0)
        # else:
        #     self.student = SwinStudentTiny(pretrained=False, num_classes=0)
        #     weights_ok = _load_student_from_checkpoint(self.student, student_checkpoint)
        #     if not weights_ok:
        #         print("[depth_model] NaN detected — falling back to timm pretrained.")
        #         self.student = SwinStudentTiny(pretrained=True, num_classes=0)
        # ─────────────────────────────────────────────────────────────────────

        self.freeze_student = freeze_student
        if freeze_student:
            for p in self.student.parameters():
                p.requires_grad = False
            self.student.eval()

        # DPT depth head — trainable
        self.dpt_head = DPTDepthHead(pretrained=dpt_pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised input images.

        Returns:
            depth: (B, 1, H, W) predicted depth map, upsampled to input size.
        """
        # Student backbone produces 4 NCHW feature maps.
        # Run under no_grad (student is frozen — requires_grad=False on all params),
        # then DETACH so the DPT head's backward graph starts at the feature tensors
        # rather than trying to trace back through the frozen student.
        #
        # IMPORTANT: do NOT use `torch.set_grad_enabled(self.student.training)` here.
        # When freeze_student=True the student is always in eval mode, so
        # `self.student.training` is always False, which would silently wrap the whole
        # forward — including the DPT-head forward — in a no-grad context, severing
        # all gradients to the DPT neck convolutions.
        with torch.no_grad():
            features, _ = self.student(x)

        # Detach from the no-grad computation graph so DPT-head autograd is clean.
        features = [f.detach() for f in features]

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
