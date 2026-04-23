"""
Convert the NYU-Depth V2 .mat file into the flat directory layout expected
by data/nyu_depth_dataset.py:

    <out_dir>/
        train/
            rgb/    0000.jpg  0001.jpg  ...
            depth/  0000.png  0001.png  ...   (16-bit, value / 256 = metres)
        val/
            rgb/    ...
            depth/  ...

The official split uses the last 215 scenes for validation (indices 1200–1449).
A common train/val split is 654 val images (standard Eigen split) — this script
uses the simpler 80/20 split by default.

Usage:
    python scripts/download_nyu_depth.py \
        --mat   nyu_depth_v2_labeled.mat \
        --out   data/nyu_depth_v2 \
        --val_fraction 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def convert(mat_path: str, out_dir: str, val_fraction: float = 0.1) -> None:
    try:
        import h5py
    except ImportError:
        raise ImportError("pip install h5py")

    mat_path = Path(mat_path)
    out_dir  = Path(out_dir)

    print(f"Opening {mat_path} …")
    with h5py.File(mat_path, "r") as f:
        images = f["images"]   # (N, 3, H, W)  uint8
        depths = f["depths"]   # (N, H, W)      float32, metres

        N = images.shape[0]
        n_val   = max(1, int(N * val_fraction))
        n_train = N - n_val
        print(f"Total samples: {N}  |  train: {n_train}  |  val: {n_val}")

        splits = {
            "train": range(0, n_train),
            "val":   range(n_train, N),
        }

        for split, indices in splits.items():
            rgb_dir   = out_dir / split / "rgb"
            depth_dir = out_dir / split / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            print(f"Writing {split} ({len(indices)} samples) …")
            for i, idx in enumerate(indices):
                # images stored as (3, H, W) — transpose to (H, W, 3)
                rgb_np = images[idx].transpose(1, 2, 0).astype(np.uint8)
                # NYU images are stored mirrored
                rgb_np = np.fliplr(rgb_np)

                depth_np = depths[idx]               # (H, W) float32 metres
                depth_np = np.fliplr(depth_np)
                # Encode as 16-bit PNG: value / 256 = metres  ↔  value = metres * 256
                depth_u16 = (depth_np * 256.0).clip(0, 65535).astype(np.uint16)

                stem = f"{i:04d}"
                Image.fromarray(rgb_np).save(rgb_dir / f"{stem}.jpg", quality=95)
                Image.fromarray(depth_u16, mode="I;16").save(depth_dir / f"{stem}.png")

                if (i + 1) % 100 == 0:
                    print(f"  {split}: {i+1}/{len(indices)}")

    print(f"\nDone. Dataset written to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mat",          default="nyu_depth_v2_labeled.mat")
    p.add_argument("--out",          default="data/nyu_depth_v2")
    p.add_argument("--val_fraction", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.mat, args.out, args.val_fraction)
