#!/bin/bash
#SBATCH --job-name=distill_sweep
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ag11023@nyu.edu
#SBATCH --array=0-11%4         # 12 combinations, run 4 concurrently

# =============================================================================
# sweep.s  —  SLURM array job for hyperparameter sweeps
#
# Each array task picks a different combination of hyperparameters from
# SWEEP_CONFIGS below and launches train.py with --override.
#
# Usage:
#   sbatch scripts/sweep.s                  # full 9-run sweep
#   sbatch --array=0-2 scripts/sweep.s      # just the first 3
#
# Add / remove entries in SWEEP_CONFIGS to change what gets swept.
# Format: each entry is a space-separated list of KEY=VALUE overrides
#         that will be passed directly to train.py --override.
# =============================================================================

set -euo pipefail

REPO=/scratch/ag11023/HPML/Efficient-Distillation
SIF=/scratch/ag11023/SIF/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif
OVERLAY=/scratch/$USER/images/Efficient-Distillation/overlay-15GB-500K.ext3

mkdir -p "$REPO/logs"

# ---------------------------------------------------------------------------
# Sweep grid — loss weight sweep
# Varying: w_kd, w_task, w_feat, w_at
# Baseline from literature: w_kd=1.0, w_task=0.1 (Hinton / TinyViT recipe)
# feat_loss_type and temperature held fixed (cosine, τ=4.0)
# ---------------------------------------------------------------------------
SWEEP_CONFIGS=(
    # # --- Baseline: all equal (current config, sanity reference) ---
    # "w_kd=1.0 w_task=1.0 w_feat=1.0 w_at=1.0 epochs=20"

    # # --- Hinton-style: KD dominant, task as light anchor ---
    # "w_kd=1.0 w_task=0.1 w_feat=1.0 w_at=1.0 epochs=20"


    # # --- KD only: zero out feature and attention losses ---
    # "w_kd=1.0 w_task=0.1 w_feat=0.0 w_at=0.0 epochs=20"

    # # --- Feature-heavy: trust intermediate representations more ---
    # "w_kd=1.0 w_task=0.1 w_feat=2.0 w_at=1.0 epochs=20"

    # # --- AT-heavy: attention alignment as main structural signal ---
    # "w_kd=1.0 w_task=0.1 w_feat=0.5 w_at=2.0 epochs=20"

    # # --- Balanced distillation (feat + AT + KD equally, no task) ---
    # "w_kd=1.0 w_task=0.0 w_feat=1.0 w_at=1.0 epochs=20"

    # # --- TinyViT-inspired: KD dominant, light feat, no AT ---
    # "w_kd=0.9 w_task=0.1 w_feat=0.3 w_at=0.0 epochs=20"

    # # --- Aggressive task suppression with feat alignment ---
    # "w_kd=1.0 w_task=0.05 w_feat=1.0 w_at=0.5 epochs=20"

    # --- Logit-only variants: how much does w_task hurt? ---
    "w_kd=1.0 w_task=0.5 w_feat=0.5 w_at=0.5 epochs=20"

    # -- Task only (no distillation) -- #
    "w_kd=0.0 w_task=0.8 w_feat=0.0 w_at=0.0 epochs=20"

    # # --- Hinton-style: KD dominant, task as light anchor ---
    # "w_kd=0.9 w_task=0.1 w_feat=0.5 w_at=0.5 epochs=20"

    # # --- Feature-heavy: trust intermediate representations more ---
    # "w_kd=1.0 w_task=0.1 w_feat=2.0 w_at=0.0 epochs=20"

    # # --- Logit-only variants: how much does w_task hurt? --- 
    # "w_kd=1.0 w_task=0.0 w_feat=0.5 w_at=0.5 epochs=20"

    
)

OVERRIDES="${SWEEP_CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Each task gets its own checkpoint directory — prevents concurrent jobs
# from clobbering each other's best.pth / epoch_NNN.pth files.
TASK_SAVE_DIR="checkpoints/sweep_${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
OVERRIDES="$OVERRIDES save_dir=$TASK_SAVE_DIR"

echo "Job:       $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Overrides: $OVERRIDES"
echo "Save dir:  $TASK_SAVE_DIR"
echo "Started:   $(date)"
echo "------------------------------------------------------------"

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt:ro \
    "$SIF" \
/bin/bash << SINGULARITY_EOF
set -euo pipefail

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

export UV_INSTALL_DIR=/scratch/ag11023/uv/bin
export UV_DATA_DIR=/scratch/ag11023/uv/data
export UV_CACHE_DIR=/scratch/ag11023/uv/cache
export UV_PYTHON_INSTALL_DIR=/scratch/ag11023/uv/python
export PATH="/scratch/ag11023/uv/bin:\$PATH"

# Redirect triton + inductor kernel caches away from /home (quota limited)
# to /scratch which has no quota issues.
export TRITON_CACHE_DIR=/scratch/ag11023/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/ag11023/inductor_cache
mkdir -p "\$TRITON_CACHE_DIR" "\$TORCHINDUCTOR_CACHE_DIR"

source $REPO/.venv/bin/activate

echo "Python:  \$(python --version)"
echo "torch:   \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    \$(python -c 'import torch; print(torch.version.cuda)')"
echo "------------------------------------------------------------"

cd $REPO

python train.py \
    --config configs/distill_config.yaml \
    --override $OVERRIDES

echo "------------------------------------------------------------"
echo "Sweep task $SLURM_ARRAY_TASK_ID finished: \$(date)"
SINGULARITY_EOF
