#!/bin/bash
#SBATCH --job-name=distill_smoke2
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --output=logs/smoke2epoch_%j.out
#SBATCH --error=logs/smoke2epoch_%j.err

# =============================================================================
# smoke2epoch.s — 2-epoch smoke test: full data pipeline, wandb enabled,
#                 checkpointing on, real batch size. Good for confirming
#                 loss curves and W&B logging before the full run.
# Usage: sbatch scripts/smoke2epoch.s
# =============================================================================

set -euo pipefail

REPO=/scratch/ag11023/HPML/Efficient-Distillation
SIF=/scratch/ag11023/SIF/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif
OVERLAY=/scratch/$USER/images/Efficient-Distillation/overlay-15GB-500K.ext3

mkdir -p "$REPO/logs"

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "------------------------------------------------------------"

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt:ro \
    "$SIF" \
/bin/bash << 'SINGULARITY_EOF'
set -euo pipefail

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

export UV_INSTALL_DIR=/scratch/ag11023/uv/bin
export UV_DATA_DIR=/scratch/ag11023/uv/data
export UV_CACHE_DIR=/scratch/ag11023/uv/cache
export UV_PYTHON_INSTALL_DIR=/scratch/ag11023/uv/python
export PATH="/scratch/ag11023/uv/bin:$PATH"

# Redirect triton/torch compile cache away from /home (tiny quota) to /scratch
export TRITON_CACHE_DIR=/scratch/ag11023/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/ag11023/cache/torchinductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

REPO=/scratch/ag11023/HPML/Efficient-Distillation
source $REPO/.venv/bin/activate

echo "Python:  $(python --version)"
echo "torch:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
echo "------------------------------------------------------------"

cd $REPO

# 2-epoch smoke: full batches, wandb on, checkpoints saved to checkpoints/smoke2epoch/
python train.py \
    --config configs/distill_config.yaml \
    --override \
        epochs=2 \
        batch_size=32 \
        num_workers=8 \
        log_every=50 \
        save_dir=checkpoints/smoke2epoch

echo "------------------------------------------------------------"
echo "Smoke-2epoch finished: $(date)"
SINGULARITY_EOF
