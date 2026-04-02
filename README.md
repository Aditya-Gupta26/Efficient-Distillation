# Efficient Distillation — Swin Transformer Knowledge Distillation

> NYU ECE-GY 9143 · High Performance Machine Learning · Spring 2026

Knowledge distillation from a large **Swin-Large (~197M params)** teacher down
to a lightweight **Swin-Tiny (~28M params)** student, with **feature adapters**
bridging the two models. Trained on **COCO 2017**.

---

## Architecture Overview

```
Image ──► Teacher (Swin-Large)  ──► Stage features [C=192, 384, 768, 1536]
   │                                          │
   │                                  Feature Adapter (per stage)
   │                                          │  projects student → teacher space
   └──► Student (Swin-Tiny)   ──► Stage features [C=96,  192, 384,  768 ]
                                               │
                              ┌────────────────┴──────────────────┐
                              │         Distillation Losses        │
                              │  • Feature MSE  (adapted ↔ teacher)│
                              │  • Attention Transfer              │
                              │  • Logit KL-Divergence (τ=4)       │
                              │  • Task loss (detection/seg)       │
                              └───────────────────────────────────┘
```

### Adapters

Each stage adapter is a lightweight **two-layer 1×1 Conv bottleneck**:

```
Conv2d(C_s → C_s) → BN → ReLU → Conv2d(C_s → C_t) → BN
```

This projects student features into teacher embedding space before computing
the feature distillation loss, with optional bilinear spatial alignment.

---

## Project Structure

```
Efficient-Distillation/
├── configs/
│   └── distill_config.yaml      # All hyperparameters
├── data/
│   ├── __init__.py
│   └── coco_dataset.py          # COCO DataLoader + transforms
├── distillation/
│   ├── __init__.py
│   ├── losses.py                # Feature, AT, KD, task losses
│   └── trainer.py               # Full training loop (AMP, checkpointing)
├── models/
│   ├── __init__.py
│   ├── teacher.py               # SwinTeacher (swin_large / swin_base)
│   ├── student.py               # SwinStudentTiny
│   └── adapters.py              # FeatureAdapter + SingleStageAdapter
├── scripts/
│   └── sanity_check.py          # Quick forward-pass smoke test
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py            # save / load checkpoint helpers
│   ├── logger.py                # Stdout logger
│   └── metrics.py               # Top-1/5 accuracy, COCO mAP stub
├── train.py                     # Main training entry-point
└── requirements.txt
```

---

## Setup

```bash
# 1. Create environment
conda create -n distill python=3.11 -y
conda activate distill

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### COCO Dataset

Download COCO 2017 and arrange as:

```
/data/coco/
    annotations/
        instances_train2017.json
        instances_val2017.json
    train2017/
    val2017/
```

Update `coco_root` in `configs/distill_config.yaml`.

---

## Training

```bash
# Smoke test (no COCO needed)
python scripts/sanity_check.py

# Full training
python train.py --config configs/distill_config.yaml

# Resume from checkpoint
python train.py --config configs/distill_config.yaml --resume checkpoints/epoch_010.pth
```

---

## Key Hyperparameters (`configs/distill_config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `teacher_variant` | `swin_large` | `swin_large` (~197M) or `swin_base` (~88M) |
| `adapter_stages` | `[0,1,2,3]` | Which stages get adapters |
| `w_feat` | `1.0` | Feature MSE loss weight |
| `w_at` | `0.5` | Attention Transfer loss weight |
| `w_kd` | `1.0` | Logit KD loss weight |
| `temperature` | `4.0` | KD softmax temperature |
| `lr` | `1e-4` | Peak learning rate (AdamW) |
| `epochs` | `30` | Training epochs |
| `amp` | `true` | Mixed-precision (FP16) |

---

## Parameter Counts

| Model | Total Params | Trainable |
|---|---|---|
| Swin-Large (teacher) | ~197 M | 0 (frozen) |
| Swin-Tiny (student) | ~28 M | ~28 M |
| Adapters (all 4 stages) | ~2 M | ~2 M |
