#!/bin/bash
#SBATCH --job-name=project
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ag11023@nyu.edu

# =============================================================================
# train.s  —  sbatch script for Efficient-Distillation
# Runs inside Singularity (CUDA 12.2 SIF) with a persistent uv-managed venv
# stored in the writable overlay.
#
# Usage:
#   sbatch scripts/train.s
#   sbatch scripts/train.s --export=RESUME=checkpoints/epoch_010.pth
# =============================================================================

set -euo pipefail

REPO=/scratch/ag11023/HPML/Efficient-Distillation
SIF=/scratch/ag11023/SIF/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif
OVERLAY=/scratch/$USER/images/Efficient-Distillation/overlay-15GB-500K.ext3

# Optional resume checkpoint (pass via --export=RESUME=... or set below)
RESUME="${RESUME:-}"

# Optional adapter warm-start checkpoint (pass via --export=ADAPTER_WARM_START=...)
ADAPTER_WARM_START="${ADAPTER_WARM_START:-}"

# Create log dir on the host side (outside Singularity)
mkdir -p "$REPO/logs"

echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Repo:      $REPO"
echo "Overlay:   $OVERLAY"
echo "Resume:    ${RESUME:-none}"
echo "Started:   $(date)"
echo "------------------------------------------------------------"

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt:ro \
    "$SIF" \
/bin/bash << SINGULARITY_EOF
set -euo pipefail

# ---- SSL certificates (needed for wandb / HTTPS on compute nodes) --------
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# ---- activate environment ------------------------------------------------
# .venv lives directly in the repo on /scratch (visible without overlay writes)
# Restore standard system PATH first — Singularity minimal env strips /usr/bin
# which breaks `source activate` (it calls basename, readlink, etc.)
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\$PATH"
# Redirect uv dirs to /scratch — home has an inode quota
export UV_INSTALL_DIR=/scratch/ag11023/uv/bin
export UV_DATA_DIR=/scratch/ag11023/uv/data
export UV_CACHE_DIR=/scratch/ag11023/uv/cache
export UV_PYTHON_INSTALL_DIR=/scratch/ag11023/uv/python
export PATH="/scratch/ag11023/uv/bin:\$PATH"

# Redirect triton/torch compile cache away from /home (tiny quota) to /scratch
export TRITON_CACHE_DIR=/scratch/ag11023/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/ag11023/cache/torchinductor
mkdir -p "\$TRITON_CACHE_DIR" "\$TORCHINDUCTOR_CACHE_DIR"

source $REPO/.venv/bin/activate

echo "Python:  \$(python --version)"
echo "torch:   \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    \$(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     \$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")')"
echo "------------------------------------------------------------"

cd $REPO

# ---- optional resume / warm-start args ----------------------------------
RESUME_ARG=""
if [ -n "${RESUME}" ]; then
    RESUME_ARG="--resume ${RESUME}"
    echo "Resuming from: ${RESUME}"
fi

WARM_START_ARG=""
if [ -n "${ADAPTER_WARM_START}" ]; then
    WARM_START_ARG="--adapter_warm_start ${ADAPTER_WARM_START}"
    echo "Adapter warm-start from: ${ADAPTER_WARM_START}"
fi

# ---- launch training ------------------------------------------------------
python train.py \
    --config configs/depth_config.yaml \
    \$RESUME_ARG \
    \$WARM_START_ARG

echo "------------------------------------------------------------"
echo "Training finished: \$(date)"
SINGULARITY_EOF
