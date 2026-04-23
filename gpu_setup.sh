#!/usr/bin/env bash
# =============================================================================
# gpu_setup.sh — ONE-TIME remote GPU setup
# =============================================================================
# Run this exactly once from your local Mac.  It will:
#   1. Copy your SSH public key to the remote so you never type a password again
#   2. Install system packages (tmux, python3-venv) on the remote
#   3. Create the project directory and a Python venv on the remote
#   4. Install PyTorch (CUDA 12.6) + all project requirements
#   5. Sync the local data/ directory to the remote (312 MB, one-time transfer)
#
# Usage:
#   chmod +x gpu_setup.sh
#   ./gpu_setup.sh
# =============================================================================

set -euo pipefail

# ── Remote config ─────────────────────────────────────────────────────────────
REMOTE_USER="newuser"
REMOTE_HOST="100.37.41.165"
REMOTE_DIR="/home/newuser/Efficient-Distillation"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
SUDO_PASS="password"        # sudo password on the remote machine

echo "============================================================"
echo "  Remote GPU Setup"
echo "  Target: ${REMOTE}:${REMOTE_DIR}"
echo "============================================================"
echo ""

# ── Step 1: Copy SSH public key so future connections are passwordless ─────────
echo "[1/5] Setting up passwordless SSH (you will be asked for your password ONCE) ..."
# Generate an SSH key pair if none exists yet
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "  No SSH key found — generating one now ..."
    ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
fi
ssh-copy-id -i ~/.ssh/id_ed25519.pub "${REMOTE}"
echo "  ✓ SSH key installed. You will not need to type a password again."
echo ""

# ── Step 2: Install system packages on remote ─────────────────────────────────
echo "[2/5] Installing system packages on remote (tmux, python3-venv) ..."
ssh "${REMOTE}" "echo '${SUDO_PASS}' | sudo -S apt-get update -qq && echo '${SUDO_PASS}' | sudo -S apt-get install -y tmux python3-venv python3-pip"
echo "  ✓ System packages ready."
echo ""

# ── Step 3: Create project directory and Python venv ──────────────────────────
echo "[3/5] Creating project directory and Python virtual environment ..."
ssh "${REMOTE}" "
    mkdir -p ${REMOTE_DIR}
    cd ${REMOTE_DIR}
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip --quiet
"
echo "  ✓ Virtual environment created at ${REMOTE_DIR}/.venv"
echo ""

# ── Step 4: Install PyTorch (CUDA) + project requirements ─────────────────────
echo "[4/5] Installing PyTorch with CUDA 12.6 + project dependencies ..."
echo "  (This takes 3-5 minutes — PyTorch is ~2 GB)"
echo ""

ssh "${REMOTE}" "
    cd ${REMOTE_DIR}

    # RTX 5090 is Blackwell (sm_120) — requires PyTorch nightly with CUDA 12.8.
    # Stable PyTorch only supports up to sm_90 (Ada/Hopper); using nightly is mandatory.
    .venv/bin/pip install --pre torch torchvision \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        --quiet

    # Project requirements from requirements.txt
    .venv/bin/pip install \
        timm>=0.9.12 \
        transformers>=4.40.0 \
        numpy>=1.24 \
        Pillow \
        matplotlib \
        tqdm>=4.66 \
        pyyaml>=6.0 \
        wandb>=0.17.0 \
        --quiet

    # Confirm GPU is visible
    echo ''
    echo 'PyTorch CUDA check:'
    .venv/bin/python -c \"
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU             : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
\"
"
echo ""
echo "  ✓ All Python dependencies installed."
echo ""

# ── Step 5: Sync the data directory to remote (one-time, 312 MB) ──────────────
echo "[5/5] Syncing data/ to remote (312 MB — this takes ~1 minute on a good connection) ..."
rsync -avz --progress \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    data/ "${REMOTE}:${REMOTE_DIR}/data/"
echo ""
echo "  ✓ Data synced."
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  You can now run any command on the GPU with:"
echo "    ./gpu_run.sh python finetuning/frozenBase_customHead.py \\"
echo "        --student_checkpoint none \\"
echo "        --nyu_root data/nyu_depth_v2 \\"
echo "        --epochs 30 \\"
echo "        --save_dir checkpoints/depth_head"
echo "============================================================"
