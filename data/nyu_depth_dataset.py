"""
NYU-Depth V2 dataset loader.

Expected directory layout (produced by scripts/download_nyu_depth.py):
    <root>/
        train/
            rgb/     *.jpg
            depth/   *.png   (16-bit, depth_meters = pixel / 256.0)
        val/
            rgb/     *.jpg
            depth/   *.png

Batch format:
    {
        "images": Tensor(B, 3, H, W)   float32, ImageNet-normalised
        "depths": Tensor(B, H, W)      float32, metres  (0–10 m)
    }
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


# ImageNet stats used during student pre-training — keep consistent.
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

NYU_MAX_DEPTH = 10.0  # metres


class NyuDepthDataset(Dataset):
    """
    Loads paired RGB + depth from the flat file layout written by the
    download script.  Depth PNGs are 16-bit; stored value / 256 = metres.
    """

    def __init__(self, root: str, split: str, img_size: int = 224):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        self.split    = split
        self.img_size = img_size

        rgb_dir   = Path(root) / split / "rgb"
        depth_dir = Path(root) / split / "depth"

        if not rgb_dir.exists():
            raise FileNotFoundError(
                f"RGB directory not found: {rgb_dir}\n"
                "Run scripts/download_nyu_depth.py first."
            )

        self.rgb_paths   = sorted(rgb_dir.glob("*.jpg"))
        self.depth_paths = sorted(depth_dir.glob("*.png"))

        if len(self.rgb_paths) != len(self.depth_paths):
            raise RuntimeError(
                f"RGB/depth count mismatch: {len(self.rgb_paths)} vs {len(self.depth_paths)}"
            )
        if len(self.rgb_paths) == 0:
            raise RuntimeError(f"No images found in {rgb_dir}")

        self.normalize = transforms.Normalize(mean=_MEAN, std=_STD)

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int) -> dict:
        rgb   = Image.open(self.rgb_paths[idx]).convert("RGB")
        depth = Image.open(self.depth_paths[idx])  # mode 'I;16' or 'I'

        if self.split == "train":
            rgb, depth = self._train_transform(rgb, depth)
        else:
            rgb, depth = self._val_transform(rgb, depth)

        # Depth: stored uint16 → metres
        depth_np = np.array(depth, dtype=np.float32) / 256.0
        depth_t  = torch.from_numpy(depth_np)
        # Clamp to valid sensor range
        depth_t  = depth_t.clamp(0.0, NYU_MAX_DEPTH)

        # RGB: to tensor + normalise
        rgb_t = TF.to_tensor(rgb)
        rgb_t = self.normalize(rgb_t)

        return {"images": rgb_t, "depths": depth_t}

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _train_transform(self, rgb: Image.Image, depth: Image.Image):
        # Random horizontal flip (same for both)
        if torch.rand(1) > 0.5:
            rgb   = TF.hflip(rgb)
            depth = TF.hflip(depth)

        # Random crop to img_size
        i, j, h, w = transforms.RandomCrop.get_params(
            rgb, output_size=(self.img_size, self.img_size)
        )
        rgb   = TF.crop(rgb,   i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)

        # Color jitter on RGB only
        rgb = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(rgb)

        return rgb, depth

    def _val_transform(self, rgb: Image.Image, depth: Image.Image):
        # Centre crop to img_size
        rgb   = TF.center_crop(rgb,   self.img_size)
        depth = TF.center_crop(depth, self.img_size)
        return rgb, depth


def build_nyu_depth_dataloaders(
    root:        str,
    img_size:    int  = 224,
    batch_size:  int  = 16,
    num_workers: int  = 8,
    pin_memory:  bool = True,
) -> tuple[DataLoader, DataLoader]:

    train_ds = NyuDepthDataset(root, split="train", img_size=img_size)
    val_ds   = NyuDepthDataset(root, split="val",   img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )

    return train_loader, val_loader
