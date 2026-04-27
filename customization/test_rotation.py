"""
test_rotation.py — Verify the 90° CCW rotation looks correct on NYU images.

Loads 3 images from the local val set and saves a side-by-side PNG:
  Left  : original (sideways, as stored on disk)
  Right : rotated 90° CCW (should be upright)

Run from the project root:
    python customization/test_rotation.py

Output: test_rotation_check.png  (opens it automatically if possible)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.nyu_depth_dataset import NyuDepthDataset

NYU_ROOT  = "data/nyu_depth_v2"
OUT_PATH  = "test_rotation_check.png"
N_SAMPLES = 3   # number of images to show

_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

def to_rgb(tensor):
    img = (tensor.numpy() * _STD + _MEAN).clip(0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)

dataset = NyuDepthDataset(root=NYU_ROOT, split="val", img_size=224)
indices = [0, 10, 50]   # fixed — no randomness

fig, axes = plt.subplots(N_SAMPLES, 4, figsize=(12, 3 * N_SAMPLES))
fig.suptitle("Rotation check — Left: original  |  Right: rotated 90° CCW",
             fontsize=12, fontweight="bold")

col_titles = ["RGB (original)", "RGB (rotated)", "Depth GT (original)", "Depth GT (rotated)"]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=9, fontweight="bold")

for row, idx in enumerate(indices):
    sample  = dataset[idx]
    rgb     = to_rgb(sample["images"])
    depth   = sample["depths"].numpy()

    axes[row, 0].imshow(rgb);                                    axes[row, 0].set_ylabel(f"idx {idx}", fontsize=8); axes[row, 0].axis("off")
    axes[row, 1].imshow(np.rot90(rgb,   k=1));                   axes[row, 1].axis("off")
    axes[row, 2].imshow(depth,          cmap="magma");           axes[row, 2].axis("off")
    axes[row, 3].imshow(np.rot90(depth, k=1), cmap="magma");     axes[row, 3].axis("off")

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_PATH}")

# Try to open it automatically
import subprocess, platform
try:
    if platform.system() == "Darwin":
        subprocess.run(["open", OUT_PATH], check=False)
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", OUT_PATH], check=False)
except Exception:
    pass
