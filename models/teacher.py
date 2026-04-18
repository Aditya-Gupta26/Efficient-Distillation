"""
Teacher Model: Swin-Base (~88M) or Swin-Large (~197M) pretrained backbone.

We wrap a timm Swin Transformer so that it exposes intermediate stage
feature maps needed for feature-level knowledge distillation.

Swin-Large output channels per stage:
    Stage 0  (1/4  resolution) : 192
    Stage 1  (1/8  resolution) : 384
    Stage 2  (1/16 resolution) : 768
    Stage 3  (1/32 resolution) : 1536

Swin-Base output channels per stage:
    Stage 0 : 128
    Stage 1 : 256
    Stage 2 : 512
    Stage 3 : 1024
"""

import torch
import torch.nn as nn
from timm import create_model


# Map of supported teacher variants → timm model names
# swin_large: use the 22K→1K fine-tuned checkpoint so the pretrained head is Linear(1536→1000)
# swin_base:  base 1K checkpoint already has Linear(1024→1000) natively
TEACHER_VARIANTS = {
    "swin_large": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",  # ~197 M, head: 1536→1000
    "swin_base":  "swin_base_patch4_window7_224",                     # ~88  M, head: 1024→1000
}

# Channel widths emitted at each of the 4 stages
TEACHER_STAGE_CHANNELS = {
    "swin_large": [192, 384, 768, 1536],
    "swin_base":  [128, 256, 512, 1024],
}


class SwinTeacher(nn.Module):
    """
    Swin-Transformer teacher that returns both the final logits and a list
    of intermediate feature maps from each hierarchical stage.

    Args:
        variant (str): One of ``"swin_large"`` (default, ~197 M) or
            ``"swin_base"`` (~88 M).
        pretrained (bool): Load ImageNet-22K pretrained weights via timm.
        num_classes (int): Number of output classes (ImageNet-1K = 1000).
        frozen_stages (int): Freeze the first N stages (0 = nothing frozen).
    """

    def __init__(
        self,
        variant: str = "swin_large",
        pretrained: bool = True,
        num_classes: int = 1000,
        frozen_stages: int = 2,
    ):
        super().__init__()
        assert variant in TEACHER_VARIANTS, (
            f"Unknown teacher variant '{variant}'. "
            f"Choose from {list(TEACHER_VARIANTS.keys())}."
        )
        self.variant = variant
        self.stage_channels = TEACHER_STAGE_CHANNELS[variant]
        self.num_classes = num_classes

        # Single model — one forward pass produces both intermediate features
        # AND the final logits via timm's forward_intermediates().
        # This halves memory vs. keeping a separate features_only backbone.
        self.model = create_model(
            TEACHER_VARIANTS[variant],
            pretrained=pretrained,
        )

        # Freeze everything — teacher is never trained
        for p in self.model.parameters():
            p.requires_grad = False

        # Optionally unfreeze later stages for fine-tuning (currently all frozen)
        # _freeze_stages is a no-op here since all params are already frozen,
        # but kept for API compatibility.
        self._frozen_stages = frozen_stages

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Image tensor of shape (B, 3, H, W).

        Returns:
            features (list[Tensor]): Per-stage feature maps
                [(B, C_i, H_i, W_i) for i in 0..3].
            logits (Tensor | None): Classification logits (B, num_classes).
        """
        # forward_intermediates returns (intermediates, final_output).
        # intermediates is a list of stage outputs in channels-last (B, H, W, C).
        # final_output is the pre-head pooled feature — we still need a full
        # forward for logits, so we call the model twice only when needed...
        # Actually timm's forward_intermediates does NOT run the head.
        # So: get features from forward_intermediates, logits from model(x).
        # Both share the same backbone weights — no duplication in memory.

        # forward_intermediates returns (final_features, intermediates_list).
        # intermediates_list contains per-stage outputs in NCHW format.
        _, features = self.model.forward_intermediates(
            x,
            indices=[0, 1, 2, 3],   # all 4 Swin stages
            output_fmt="NCHW",       # get (B, C, H, W) directly — no permute needed
        )

        logits = None
        if self.num_classes > 0:
            logits = self.model(x)   # full forward through intact pretrained head

        return features, logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _freeze_stages(self, num_stages: int) -> None:
        """No-op: all teacher params are frozen at init. Kept for API compat."""
        pass

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (e.g., for fine-tuning the teacher)."""
        for p in self.model.parameters():
            p.requires_grad = True

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
