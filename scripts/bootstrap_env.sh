#!/bin/bash
# =============================================================================
# bootstrap_env.sh
# One-time setup: installs uv + creates .venv directly in the repo on /scratch.
# No overlay writes needed — /scratch is bind-mounted inside Singularity.
#
# Run this ONCE from a login node:
#   bash scripts/bootstrap_env.sh
# =============================================================================

set -euo pipefail

REPO=/scratch/ag11023/HPML/Efficient-Distillation
SIF=/scratch/ag11023/SIF/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif
OVERLAY=/scratch/$USER/images/Efficient-Distillation/overlay-15GB-500K.ext3

echo ">>> Bootstrapping environment inside Singularity..."
echo "    SIF:     $SIF"
echo "    Repo:    $REPO"
echo "    Venv:    $REPO/.venv  (lives on /scratch, no overlay writes needed)"
echo ""

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "$SIF" \
/bin/bash << 'EOF'
set -euo pipefail

REPO=/scratch/ag11023/HPML/Efficient-Distillation
cd "$REPO"

# ------------------------------------------------------------------
# Redirect ALL uv dirs to /scratch — home has an inode quota
# ------------------------------------------------------------------
export UV_INSTALL_DIR=/scratch/ag11023/uv/bin
export UV_DATA_DIR=/scratch/ag11023/uv/data
export UV_CACHE_DIR=/scratch/ag11023/uv/cache
export UV_PYTHON_INSTALL_DIR=/scratch/ag11023/uv/python
export PATH="/scratch/ag11023/uv/bin:$PATH"

# ------------------------------------------------------------------
# 1. Install uv into /scratch/ag11023/uv/bin
# ------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv to $UV_INSTALL_DIR ..."
    mkdir -p "$UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo ">>> uv already installed: $(uv --version)"
fi

# ------------------------------------------------------------------
# 2. Create .venv directly inside the repo on /scratch
#    (same pattern as PufferDrive — scratch is bind-mounted in Singularity)
# ------------------------------------------------------------------
if [ ! -d "$REPO/.venv" ]; then
    echo ">>> Creating $REPO/.venv with Python 3.11..."
    uv venv "$REPO/.venv" --python 3.11
else
    echo ">>> .venv already exists at $REPO/.venv"
fi

source "$REPO/.venv/bin/activate"
echo ">>> Python: $(python --version) at $(which python)"

# ------------------------------------------------------------------
# 3. Install PyTorch — cu121 wheels match the cuda12.2 SIF
# ------------------------------------------------------------------
echo ">>> Installing PyTorch (cu121)..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------------
# 4. Install project requirements
# ------------------------------------------------------------------
echo ">>> Installing project dependencies..."
uv pip install -r "$REPO/requirements.txt"

# ------------------------------------------------------------------
# 5. Smoke test (no GPU on login node, skip CUDA check)
# ------------------------------------------------------------------
echo ">>> Smoke test..."
python -c "import torch;       print('torch', torch.__version__)"
python -c "import timm;        print('timm', timm.__version__)"
python -c "import pycocotools; print('pycocotools OK')"

echo ""
echo "============================================"
echo "  Bootstrap complete!"
echo "  Activate inside Singularity with:"
echo "    source $REPO/.venv/bin/activate"
echo "============================================"
EOF
