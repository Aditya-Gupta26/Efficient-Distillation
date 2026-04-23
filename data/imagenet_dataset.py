"""
ImageNet-1K DataLoaders for Knowledge Distillation.

Expected directory layout:
    <root>/
        train/
            n01440764/   (synset folders)
            n01443537/
            ...
        val/
            n01440764/
            n01443537/
            ...

Uses torchvision.datasets.ImageFolder — each subfolder is a class.
Returns batches of:
    {
        "images":  (B, 3, H, W)  float32 tensor,
        "targets": (B,)          int64 tensor  (class index 0-999)
    }
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as T
import os

try:
    from timm.data import ImageNetInfo
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Transforms  (standard ImageNet recipe)
# ---------------------------------------------------------------------------

def build_train_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_val_transforms(img_size: int = 224) -> T.Compose:
    # Standard: resize to 256, centre-crop to img_size
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Label remapping: ImageFolder sorts by folder name alphabetically.
# If folders use human-readable names ("Afghan hound, Afghan") instead of
# synset IDs (n02088094), the alphabetical order differs from the canonical
# ImageNet 0-999 ordering the model was trained with.
# This builds a remap tensor: imagefolder_idx → canonical model idx.
# ---------------------------------------------------------------------------

def build_label_remap(dataset_dir: str) -> torch.Tensor | None:
    """Return a remap tensor or None if timm is unavailable / not needed."""
    if not _TIMM_AVAILABLE:
        return None
    folder_classes = sorted(os.listdir(dataset_dir))
    # If folders are already synset IDs (n0XXXXXXX), no remap needed
    if folder_classes[0].startswith("n0"):
        return None
    try:
        info = ImageNetInfo("imagenet-1k")
        synsets = info.label_names()  # list of 1000 synset IDs in canonical order
        # Build lemma → canonical index (folder names use lemma strings)
        lemma_to_idx: dict[str, int] = {
            info.label_name_to_description(synset): canonical_idx
            for canonical_idx, synset in enumerate(synsets)
        }
        # One-off fix: two "crane" classes share the lemma "crane" in timm.
        # 'crane2' folder = bird crane (canonical idx 134).
        lemma_to_idx["crane2"] = 134
        remap = torch.zeros(len(folder_classes), dtype=torch.long)
        for i, name in enumerate(folder_classes):
            remap[i] = lemma_to_idx.get(name, i)  # identity fallback if not found
        return remap
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Collation: return a clean dict matching the trainer's expected interface
# ---------------------------------------------------------------------------

def make_collate_fn(remap: torch.Tensor | None):
    """Build a collate_fn that optionally remaps targets to canonical indices."""
    def collate_fn(batch):
        images, targets = zip(*batch)
        t = torch.tensor(targets, dtype=torch.int64)
        if remap is not None:
            t = remap[t]
        return {"images": torch.stack(images, dim=0), "targets": t}
    return collate_fn


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_imagenet_dataloaders(
    imagenet_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build ImageNet-1K train and val DataLoaders.

    Args:
        imagenet_root: Path to ImageNet root containing ``train/`` and ``val/``.
        img_size     : Spatial size to resize/crop images to.
        batch_size   : Batch size per GPU.
        num_workers  : Number of DataLoader worker processes.
        pin_memory   : Whether to pin memory (recommended with CUDA).

    Returns:
        (train_loader, val_loader)
    """
    train_dir = os.path.join(imagenet_root, "train")
    val_dir   = os.path.join(imagenet_root, "val")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=build_train_transforms(img_size),
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=build_val_transforms(img_size),
    )

    # Build label remap in case folders use human-readable names
    remap = build_label_remap(train_dir)
    if remap is not None:
        print(f"[imagenet_dataset] Label remap applied "
              f"(folder names → canonical ImageNet indices). "
              f"e.g. folder[0] → class {remap[0].item()}")
    collate_fn = make_collate_fn(remap)

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        if distributed
        else None
    )

    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
