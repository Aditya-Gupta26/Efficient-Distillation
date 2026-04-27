#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh — Run all 5 experiments in sequence then compare.
# =============================================================================
# This script is designed to be launched once via gpu_run.sh and left to run
# unattended.  Track progress live in Weights & Biases.
#
# Order of execution:
#   0. Intel DPT baseline   (inference only — no training)
#   1. frozenBase_pretrainedUnfrozenHead
#   2. unfrozenBase_pretrainedUnfrozenHead
#   3. frozenBase_unfrozenHead
#   4. unfrozenBase_unfrozenHead
#   5. validation.py        (compare all models, produces metric table + grid)
#
# Usage:
#   # 30-epoch test run (default):
#   bash customization/run_all_experiments.sh
#
#   # Full 300-epoch run (resumes from 30-epoch checkpoints automatically):
#   bash customization/run_all_experiments.sh --epochs 300
#
# That's it — everything else (W&B key, project, paths) is hardcoded above.
# =============================================================================

set -euo pipefail

# ── Configuration — only edit this block ──────────────────────────────────────
EPOCHS=30                    # default; override with:  --epochs 300
STUDENT="checkpoints/best.pth"
NYU_ROOT="data/nyu_depth_v2"
WANDB_PROJECT="efficient-distillation"
WANDB_ENTITY="ag11023-new-york-university"  # W&B team entity (not username)
WANDB_API_KEY="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"

# Authenticate every child Python process automatically
export WANDB_API_KEY

# ── Parse arguments — only --epochs is accepted ───────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)  EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1  (only --epochs is accepted)"; exit 1 ;;
    esac
done

PYTHON="$(pwd)/.venv/bin/python"

# Shared flags passed to every training script
COMMON="--student_checkpoint ${STUDENT} --nyu_root ${NYU_ROOT} --epochs ${EPOCHS} \
        --wandb_project ${WANDB_PROJECT} --wandb_entity ${WANDB_ENTITY}"

# ── Print banner ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          NYU Depth Estimation — Full Experiment Suite        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Student checkpoint : ${STUDENT}"
echo "║  Epochs per run     : ${EPOCHS}"
echo "║  NYU root           : ${NYU_ROOT}"
echo "║  W&B project        : ${WANDB_PROJECT}"
echo "║  W&B API key        : ${WANDB_API_KEY:+set (${#WANDB_API_KEY} chars)}${WANDB_API_KEY:-NOT SET — W&B logging will be disabled}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Helper: print a section banner ───────────────────────────────────────────
step() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP $1: $2"
    echo "  Started: $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Intel DPT baseline (no training — just inference + metrics)
# ─────────────────────────────────────────────────────────────────────────────
step 0 "Intel/dpt-swinv2-tiny-256 baseline evaluation"
$PYTHON customization/Intel_dpt_swin_tiny_256.py \
    --nyu_root "${NYU_ROOT}" \
    --results "checkpoints/customization/intel_dpt_results.json"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Frozen student + Pretrained DPT head
# ─────────────────────────────────────────────────────────────────────────────
step 1 "frozenBase_pretrainedUnfrozenHead  (frozen student + pretrained head)"
$PYTHON customization/frozenBase_pretrainedUnfrozenHead.py \
    $COMMON \
    --save_dir "checkpoints/customization/frozenBase_pretrainedUnfrozenHead"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Unfrozen student + Pretrained DPT head
# ─────────────────────────────────────────────────────────────────────────────
step 2 "unfrozenBase_pretrainedUnfrozenHead  (trainable student + pretrained head)"
$PYTHON customization/unfrozenBase_pretrainedUnfrozenHead.py \
    $COMMON \
    --save_dir "checkpoints/customization/unfrozenBase_pretrainedUnfrozenHead"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Frozen student + Random DPT head
# ─────────────────────────────────────────────────────────────────────────────
step 3 "frozenBase_unfrozenHead  (frozen student + random head from scratch)"
$PYTHON customization/frozenBase_unfrozenHead.py \
    $COMMON \
    --save_dir "checkpoints/customization/frozenBase_unfrozenHead"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Unfrozen student + Random DPT head  (full end-to-end)
# ─────────────────────────────────────────────────────────────────────────────
step 4 "unfrozenBase_unfrozenHead  (trainable student + random head — full end-to-end)"
$PYTHON customization/unfrozenBase_unfrozenHead.py \
    $COMMON \
    --save_dir "checkpoints/customization/unfrozenBase_unfrozenHead"

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Compare all models
# ─────────────────────────────────────────────────────────────────────────────
step 5 "validation.py  — side-by-side comparison of all models"
$PYTHON customization/validation.py \
    --student_checkpoint "${STUDENT}" \
    --nyu_root           "${NYU_ROOT}" \
    --out_dir            "checkpoints/customization" \
    --wandb_project      "${WANDB_PROJECT}" \
    --wandb_entity       "${WANDB_ENTITY}"

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All experiments complete!                                   ║"
echo "║  Metrics  → checkpoints/customization/validation_metrics.png ║"
echo "║  Grid     → checkpoints/customization/validation_grid.png    ║"
echo "║  JSON     → checkpoints/customization/validation_metrics.json║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
