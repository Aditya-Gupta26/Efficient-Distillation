"""
Simple forward pass through the distilled Swin-Tiny student.
Picks 10 random images from NYU Depth V2, runs them through the
classification head, and prints the top-3 predicted ImageNet labels.
"""

import json
import random
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.nyu_depth_dataset import NyuDepthDataset
from models.student import SwinStudentTiny


# ── 1. Load ImageNet class names ──────────────────────────────────────────────

url = ("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
       "/master/imagenet-simple-labels.json")
with urllib.request.urlopen(url, timeout=5) as r:
    labels = json.loads(r.read().decode())   # list of 1000 plain-English names


# ── 2. Load the student checkpoint ────────────────────────────────────────────

CHECKPOINT = "checkpoints/epoch_069.pth"

# Build model with num_classes=1000 so the classification head stays intact
student = SwinStudentTiny(pretrained=False, num_classes=1000)

ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
state = ckpt.get("student", ckpt)                          # weights live under "student" key
state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}  # strip torch.compile prefix

# # ── Optional NaN check (uncomment to detect a corrupted / diverged checkpoint) ──
# nan_keys = [k for k, v in state.items() if torch.is_tensor(v) and torch.isnan(v).any()]
# if nan_keys:
#     print(f"CORRUPTED: {len(nan_keys)}/{len(state)} tensors are NaN.")
#     print(f"First bad keys: {nan_keys[:5]}")
#     sys.exit(1)

student.load_state_dict(state, strict=False)
student.eval()


# ── 3. Pick 10 random NYU images ──────────────────────────────────────────────

dataset = NyuDepthDataset(root="data/nyu_depth_v2", split="val", img_size=224)
random.seed(42)
indices = random.sample(range(len(dataset)), k=10)


print(f"\n{'Image':>6}   Top-3 predictions")

with torch.no_grad():
    for idx in indices:
        image = dataset[idx]["images"].unsqueeze(0)   # (1, 3, 224, 224)

        _, logits = student(image)                    # logits: (1, 1000)
        probs     = F.softmax(logits[0], dim=0)       # (1000,) probabilities
        top_probs, top_ids = probs.topk(3)

        print(f"{idx:>6}", end="   ")
        for prob, cls_id in zip(top_probs.tolist(), top_ids.tolist()):
            print(f"{labels[cls_id]} ({prob*100:.1f}%)", end="   ")
        print()
