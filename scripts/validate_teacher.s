#!/bin/bash
#SBATCH --job-name=teacher_val
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --output=logs/teacher_val_%j.out
#SBATCH --error=logs/teacher_val_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ag11023@nyu.edu

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

export TRITON_CACHE_DIR=/scratch/ag11023/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/ag11023/inductor_cache
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

source /scratch/ag11023/HPML/Efficient-Distillation/.venv/bin/activate

echo "Python: $(python --version)"
echo "torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "------------------------------------------------------------"

cd /scratch/ag11023/HPML/Efficient-Distillation

python scripts/validate_teacher.py \
    --variant swin_large \
    --num_images 100 \
    --batch_size 32

SINGULARITY_EOF

echo "------------------------------------------------------------"
echo "Done: $(date)"
