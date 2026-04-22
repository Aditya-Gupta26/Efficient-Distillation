"""
Quick teacher validation script.

Loads SwinTeacher with pretrained weights, samples N random images from the
ImageNet val set, and reports Top-1 / Top-5 accuracy against ground truth.

Usage:
    python scripts/validate_teacher.py
    python scripts/validate_teacher.py --num_images 500
    python scripts/validate_teacher.py --variant swin_base --num_images 200
"""

from __future__ import annotations

import argparse
import sys
import os

# Make sure repo root is on the path regardless of where the script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision.transforms.v2 as T
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import random

from models.teacher import SwinTeacher
from timm.data import ImageNetInfo


def parse_args():
    p = argparse.ArgumentParser(description="Validate teacher model on ImageNet val")
    p.add_argument("--variant",      default="swin_large",
                   choices=["swin_large", "swin_base"])
    p.add_argument("--imagenet_val", default="/scratch/ag11023/HPML/imagenet/val")
    p.add_argument("--num_images",   type=int, default=100)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def build_label_remap(val_root: str) -> torch.Tensor:
    """
    ImageFolder sorts folders alphabetically → indices don't match the
    canonical ImageNet 0-999 ordering the model was trained with when
    folders use human-readable names ("Afghan hound, Afghan") instead of
    synset IDs (n02088094).

    Returns a remap tensor:  imagefolder_idx → correct_model_idx
    """
    folder_classes = sorted(os.listdir(val_root))

    # If already synset IDs, no remap needed — return identity
    if folder_classes[0].startswith("n0"):
        print("  Folders are synset IDs — no label remap needed.")
        return torch.arange(len(folder_classes))

    # Use timm's built-in ImageNet-1K label map.
    # label_names() returns synset IDs (n01440764, …) in canonical 0-999 order.
    # label_name_to_description(synset) returns the human-readable lemma string
    # (e.g. "tench", "Afghan hound, Afghan") which is what the folder names use.
    info = ImageNetInfo("imagenet-1k")
    synsets = info.label_names()  # list of 1000 synset IDs, index = canonical class idx
    # Build lemma → canonical index
    lemma_to_idx: dict[str, int] = {}
    for canonical_idx, synset in enumerate(synsets):
        lemma = info.label_name_to_description(synset)
        lemma_to_idx[lemma] = canonical_idx
    # One-off fix: the dataset has two "crane" folders ('crane' and 'crane2').
    # timm maps both synsets to the lemma "crane"; 'crane2' is the bird crane (idx 134).
    lemma_to_idx["crane2"] = 134

    remap = torch.zeros(len(folder_classes), dtype=torch.long)
    n_unmapped = 0
    for i, name in enumerate(folder_classes):
        idx = lemma_to_idx.get(name)
        if idx is None:
            n_unmapped += 1
            idx = i   # identity fallback
        remap[i] = idx

    if n_unmapped:
        print(f"  Warning: {n_unmapped}/{len(folder_classes)} folders could not be remapped.")
    else:
        print(f"  All {len(folder_classes)} folders remapped successfully.")
    return remap


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Variant : {args.variant}")
    print(f"Images  : {args.num_images} random samples from {args.imagenet_val}")
    print("-" * 55)

    # ------------------------------------------------------------------
    # Build label remap  (ImageFolder alphabetical idx → model's canonical idx)
    # ------------------------------------------------------------------
    print("Building label remap...")
    remap = build_label_remap(args.imagenet_val)
    print(f"  Remap built for {len(remap)} classes. "
          f"First 5: folder[0..4] → model{remap[:5].tolist()}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("Loading SwinTeacher...")
    teacher = SwinTeacher(
        variant=args.variant,
        pretrained=True,
        num_classes=1000,
        frozen_stages=4,
    )
    teacher.eval().to(device)
    remap = remap.to(device)
    print(f"Parameters: {teacher.num_parameters:,}  |  trainable: {teacher.num_trainable_parameters:,}")

    # ------------------------------------------------------------------
    # Data — standard ImageNet val transforms
    # ------------------------------------------------------------------
    val_transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_val = datasets.ImageFolder(args.imagenet_val, transform=val_transforms)
    indices  = random.sample(range(len(full_val)), min(args.num_images, len(full_val)))
    subset   = Subset(full_val, indices)
    loader   = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    print(f"Evaluating on {len(subset)} images  ({len(loader)} batches)\n")

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------
    top1_correct = top5_correct = total = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images  = images.to(device)
            targets = targets.to(device)

            # Remap ImageFolder indices → canonical ImageNet indices
            targets = remap[targets]

            _, logits = teacher(images)   # features, logits

            # Top-1
            top1_correct += (logits.argmax(dim=1) == targets).sum().item()

            # Top-5
            top5_preds    = logits.topk(5, dim=1).indices
            top5_correct += (top5_preds == targets.unsqueeze(1)).any(dim=1).sum().item()

            total += targets.size(0)

            batch_top1 = (logits.argmax(dim=1) == targets).float().mean().item() * 100
            print(f"  Batch {batch_idx+1:>3}/{len(loader)}  "
                  f"({targets.size(0)} imgs)  batch Top-1: {batch_top1:.1f}%  "
                  f"running Top-1: {100.*top1_correct/total:.2f}%")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    print("\n" + "=" * 55)
    print(f"  Teacher : {args.variant}")
    print(f"  Samples : {total}")
    print(f"  Top-1   : {top1:.2f}%   (expected ~86% for swin_large ft_in1k)")
    print(f"  Top-5   : {top5:.2f}%   (expected ~98% for swin_large ft_in1k)")
    print("=" * 55)

    if top1 < 5.0:
        print("\n  ✗ Top-1 still near-random — pretrained weights are not loading correctly.")
    elif top1 > 50.0:
        print("\n  ✓ Pretrained weights confirmed working.")


if __name__ == "__main__":
    main()
