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
        num_classes (int): Number of output classes (COCO = 80 for detection
            head; set to 0 to return raw features only).
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

        # Build a FULL pretrained model using the DEFAULT num_classes so that
        # timm does NOT replace the pretrained head with a random one.
        # (passing num_classes != checkpoint default causes timm to re-init the head)
        full_model = create_model(
            TEACHER_VARIANTS[variant],
            pretrained=pretrained,
        )

        # Now build the features-only backbone for intermediate feature maps
        self.backbone = create_model(
            TEACHER_VARIANTS[variant],
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Transplant the pretrained head from the full model so logits are meaningful
        in_features = self.stage_channels[-1]
        if num_classes > 0:
            pretrained_head: nn.Linear | None = None
            if pretrained and hasattr(full_model, "head") and isinstance(full_model.head, nn.Linear):
                pretrained_head = full_model.head  # shape: (default_classes, in_features)

            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features, num_classes),
            )

            if pretrained_head is not None:
                if pretrained_head.out_features == num_classes:
                    # Shapes match — copy directly
                    self.head[-1].weight.data.copy_(pretrained_head.weight.data)
                    self.head[-1].bias.data.copy_(pretrained_head.bias.data)
                else:
                    raise ValueError(
                        f"Pretrained head has {pretrained_head.out_features} output classes "
                        f"but num_classes={num_classes} was requested. "
                        f"Pass num_classes={pretrained_head.out_features} to use the pretrained head."
                    )
        else:
            self.head = None

        del full_model  # free memory

        # Freeze early stages to reduce GPU memory during distillation
        self._freeze_stages(frozen_stages)

        # Freeze the head — teacher is never trained, all params must be frozen
        if self.head is not None:
            for p in self.head.parameters():
                p.requires_grad = False

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
            logits (Tensor | None): Classification logits (B, num_classes),
                or None if ``num_classes=0``.
        """
        features = self.backbone(x)          # list of 4 tensors, each (B, H, W, C)

        # timm Swin features_only uses channels-last (B, H, W, C).
        # Permute to standard (B, C, H, W) for compatibility with adapters / losses.
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        logits = None
        if self.head is not None:
            logits = self.head(features[-1])

        return features, logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _freeze_stages(self, num_stages: int) -> None:
        """Freeze patch-embed + the first `num_stages` transformer stages."""
        if num_stages < 0:
            return
        # timm features_only model exposes stages as layers_0 … layers_3
        modules_to_freeze = [self.backbone.patch_embed]
        for i in range(min(num_stages, 4)):
            modules_to_freeze.append(getattr(self.backbone, f"layers_{i}"))
        for m in modules_to_freeze:
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone parameters (e.g., for fine-tuning)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
