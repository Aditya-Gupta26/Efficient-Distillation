"""
Download and unpack NYU-Depth V2 from the official MIT source.

Source : https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
File   : nyu_depth_v2_labeled.mat  (~2.8 GB, HDF5/MATLAB v7.3 format)
Samples: 1449 labeled RGBD pairs  (480×640)
Split  : 795 train / 654 val  (standard Eigen et al. split)

Output layout (matches data/nyu_depth_dataset.py):
    <out>/
        train/
            rgb/     00000.jpg … 00794.jpg
            depth/   00000.png … 00794.png   (16-bit PNG, value/256 = metres)
        val/
            rgb/     00000.jpg … 00653.jpg
            depth/   00000.png … 00653.png

Requirements:
    pip install h5py tqdm requests pillow numpy

Usage:
    python scripts/download_nyu_depth.py --out /scratch/ag11023/HPML/nyu_depth_v2
    python scripts/download_nyu_depth.py --out /scratch/ag11023/HPML/nyu_depth_v2 --dry_run
    # If you already have the .mat file:
    python scripts/download_nyu_depth.py --mat /path/to/nyu_depth_v2_labeled.mat --out ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

MAT_URL = (
    "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/"
    "nyu_depth_v2_labeled.mat"
)

# Standard Eigen et al. split — 654 test indices (0-based, out of 1449).
# Everything else becomes training data.
# Source: https://github.com/cleinc/bts (splits/eigen_test_indices.txt)
EIGEN_TEST_INDICES = {
    1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
   14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
   27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
   40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
   53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
   66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
   79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
   92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
  105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
  118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
  131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
  144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
  157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
  170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
  183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
  196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
  209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,
  222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
  235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
  248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
  261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273,
  274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
  287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,
  300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
  313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
  326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
  339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
  352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364,
  365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
  378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390,
  391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403,
  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
  417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429,
  430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442,
  443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455,
  456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468,
  469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481,
  482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494,
  495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
  508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520,
  521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
  534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
  547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
  560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
  573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585,
  586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598,
  599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611,
  612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624,
  625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637,
  638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650,
  651, 652, 653,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/scratch/ag11023/HPML/nyu_depth_v2",
                   help="Output root directory.")
    p.add_argument("--mat", default=None,
                   help="Path to an already-downloaded .mat file (skips download).")
    p.add_argument("--dry_run", action="store_true",
                   help="Process only 20 train + 10 val samples (for testing).")
    return p.parse_args()


# ------------------------------------------------------------------ #
# Download
# ------------------------------------------------------------------ #

def download_mat(dest: Path) -> Path:
    """Download the .mat file with a progress bar. Returns local path."""
    try:
        import requests
    except ImportError:
        raise SystemExit("pip install requests")

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MAT_URL}")
    print(f"  → {dest}  (~2.8 GB, this will take a while)\n")

    with requests.get(MAT_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    return dest


# ------------------------------------------------------------------ #
# Process
# ------------------------------------------------------------------ #

def process_mat(mat_path: Path, out_root: Path, dry_run: bool) -> None:
    try:
        import h5py
    except ImportError:
        raise SystemExit("pip install h5py")

    print(f"\nOpening {mat_path} ...")
    with h5py.File(mat_path, "r") as f:
        # HDF5/MATLAB v7.3 stores arrays transposed relative to NumPy convention.
        # images : (3, W, H, N) in file  →  we read as (N, H, W, 3)
        # depths : (W, H, N)    in file  →  we read as (N, H, W)
        images_ds = f["images"]   # lazy; shape (3, 640, 480, 1449)
        depths_ds = f["depths"]   # lazy; shape (640, 480, 1449)
        N = images_ds.shape[-1]
        print(f"Total samples in .mat: {N}")

        # Build index lists
        train_idx = [i for i in range(N) if i not in EIGEN_TEST_INDICES]
        val_idx   = sorted(EIGEN_TEST_INDICES)

        if dry_run:
            train_idx = train_idx[:20]
            val_idx   = val_idx[:10]

        print(f"Split  →  train: {len(train_idx)}  |  val: {len(val_idx)}")

        _save_split(images_ds, depths_ds, train_idx, out_root / "train")
        _save_split(images_ds, depths_ds, val_idx,   out_root / "val")

    print(f"\nDone. Dataset saved to {out_root}")
    print("Verify nyu_depth_root in configs/depth_config.yaml points here.")


def _save_split(images_ds, depths_ds, indices: list, split_dir: Path) -> None:
    rgb_dir   = split_dir / "rgb"
    depth_dir = split_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    split_name = split_dir.name
    for out_idx, src_idx in enumerate(tqdm(indices, desc=split_name)):
        # images_ds shape: (3, 640, 480, N) — read one column
        # Transpose (3, W, H) → (H, W, 3) for PIL
        img_arr = images_ds[:, :, :, src_idx]      # (3, 640, 480)
        img_arr = img_arr.transpose(2, 1, 0)        # (480, 640, 3)
        Image.fromarray(img_arr.astype(np.uint8), "RGB").save(
            rgb_dir / f"{out_idx:05d}.jpg", quality=95
        )

        # depths_ds shape: (640, 480, N) — depth in metres (float32)
        d_arr = depths_ds[:, :, src_idx]            # (640, 480)
        d_arr = d_arr.T                              # (480, 640)

        # Store as uint16: value / 256 = metres  (matches nyu_depth_dataset.py)
        d_uint16 = (d_arr * 256.0).clip(0, 65535).astype(np.uint16)
        Image.fromarray(d_uint16, mode="I;16").save(
            depth_dir / f"{out_idx:05d}.png"
        )


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    args     = parse_args()
    out_root = Path(args.out)

    if args.mat:
        mat_path = Path(args.mat)
        if not mat_path.exists():
            raise SystemExit(f"--mat file not found: {mat_path}")
    else:
        mat_path = out_root / "nyu_depth_v2_labeled.mat"
        if not mat_path.exists():
            download_mat(mat_path)
        else:
            print(f"Found existing .mat file: {mat_path}")

    process_mat(mat_path, out_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
