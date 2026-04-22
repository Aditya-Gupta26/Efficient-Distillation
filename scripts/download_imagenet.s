#!/bin/bash
#SBATCH --job-name=imagenet_download
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --output=logs/imagenet_download_%j.out
#SBATCH --error=logs/imagenet_download_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ag11023@nyu.edu

# =============================================================================
# Download ImageNet-1K from HuggingFace into ImageFolder layout.
#
# Prerequisites (run ONCE on login node before submitting this job):
#   cd /scratch/ag11023/HPML/Efficient-Distillation
#   .venv/bin/python -c "from huggingface_hub import login; login()"
#   # Then accept license at: https://huggingface.co/datasets/ILSVRC/imagenet-1k
#
# Submit:
#   sbatch scripts/download_imagenet.s
# =============================================================================

set -euo pipefail

REPO=/scratch/ag11023/HPML/Efficient-Distillation
SIF=/scratch/ag11023/SIF/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif
OVERLAY=/scratch/$USER/images/Efficient-Distillation/overlay-15GB-500K.ext3
OUTPUT=/scratch/ag11023/HPML/imagenet

mkdir -p "$REPO/logs" "$OUTPUT"

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Output:  $OUTPUT"
echo "Started: $(date)"
echo "------------------------------------------------------------"

singularity exec \
    --overlay "${OVERLAY}:ro" \
    --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt:ro \
    --bind "$OUTPUT:$OUTPUT" \
    "$SIF" \
/bin/bash << SINGULARITY_EOF
set -euo pipefail

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# HuggingFace auth — token passed explicitly so gated datasets work inside Singularity
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export HUGGING_FACE_HUB_TOKEN="\$HF_TOKEN"

# Point HF cache to scratch so it doesn't blow up home inode quota
export HF_HOME=/scratch/ag11023/wandb_cache/huggingface
export HF_DATASETS_CACHE=/scratch/ag11023/wandb_cache/huggingface/datasets

export UV_INSTALL_DIR=/scratch/ag11023/uv/bin
export PATH="/scratch/ag11023/uv/bin:\$PATH"
source $REPO/.venv/bin/activate

echo "Python: \$(python --version)"

# Install datasets package into the venv if not already present
python -c "import datasets" 2>/dev/null || uv pip install datasets

cd $REPO

python scripts/download_imagenet.py \
    --output $OUTPUT \
    --num-workers 16 \
    --split both || true

echo "------------------------------------------------------------"
echo "Download finished: \$(date)"
echo "Disk usage: \$(du -sh $OUTPUT)"
SINGULARITY_EOF
