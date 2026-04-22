#!/bin/bash
# =============================================================================
# create_overlay.sh
# One-time script: create a writable ext3 overlay for Efficient-Distillation.
# Run this ONCE from a login node (no GPU needed).
#
# Usage:
#   bash scripts/create_overlay.sh
# =============================================================================

set -euo pipefail

OVERLAY_DIR="/scratch/$USER/images/Efficient-Distillation"
OVERLAY_IMG="$OVERLAY_DIR/overlay-15GB-500K.ext3"

echo ">>> Creating overlay directory: $OVERLAY_DIR"
mkdir -p "$OVERLAY_DIR"

if [ -f "$OVERLAY_IMG" ]; then
    echo "Overlay already exists at $OVERLAY_IMG — skipping creation."
    echo "Delete it and rerun if you want a fresh overlay."
    exit 0
fi

echo ">>> Copying blank overlay (from PufferDrive's clean overlay)..."
# We copy the existing 15GB ext3 overlay as a blank starting point.
# Alternatively, if NYU provides a public blank template, use that instead.
TEMPLATE=/scratch/ag11023/images/PufferDrive/overlay-15GB-500K.ext3
if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: template overlay not found at $TEMPLATE"
    echo "Provide a blank ext3 overlay manually or obtain one from your cluster docs."
    exit 1
fi
cp "$TEMPLATE" "$OVERLAY_IMG"
echo ">>> Copied overlay to $OVERLAY_IMG"

echo ">>> Overlay ready at: $OVERLAY_IMG"
echo ""
echo "Next step: run  bash scripts/bootstrap_env.sh  to install uv + dependencies."
