#!/bin/bash
#SBATCH --job-name=coco_download
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --output=logs/coco_download_%j.out
#SBATCH --error=logs/coco_download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ag11023@nyu.edu

# =============================================================================
# download_coco.s  —  Download COCO 2017 to /scratch/ag11023/HPML/coco
#
# Total size: ~25 GB
#   train2017.zip        ~18 GB   (118,287 images)
#   val2017.zip          ~1  GB   (5,000 images)
#   annotations_trainval2017.zip  ~241 MB
#
# Usage:
#   sbatch scripts/download_coco.s
# =============================================================================

set -euo pipefail

COCO_DIR=/scratch/ag11023/HPML/coco
mkdir -p "$COCO_DIR/annotations"
mkdir -p "$COCO_DIR/train2017"
mkdir -p "$COCO_DIR/val2017"

echo "Downloading COCO 2017 to $COCO_DIR"
echo "Node: $(hostname)  |  Started: $(date)"
echo "------------------------------------------------------------"

cd "$COCO_DIR"

download_if_missing() {
    local url="$1"
    local file="$2"
    if [ -f "$file" ]; then
        echo ">>> Already exists, skipping: $file"
    else
        echo ">>> Downloading $url ..."
        wget --progress=dot:giga -O "$file" "$url"
    fi
}

# ---- Download zips -------------------------------------------------------
download_if_missing \
    "http://images.cocodataset.org/zips/train2017.zip" \
    "$COCO_DIR/train2017.zip"

download_if_missing \
    "http://images.cocodataset.org/zips/val2017.zip" \
    "$COCO_DIR/val2017.zip"

download_if_missing \
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
    "$COCO_DIR/annotations_trainval2017.zip"

# ---- Extract -------------------------------------------------------------
echo ">>> Extracting train2017.zip ..."
unzip -q -n train2017.zip -d "$COCO_DIR"

echo ">>> Extracting val2017.zip ..."
unzip -q -n val2017.zip -d "$COCO_DIR"

echo ">>> Extracting annotations_trainval2017.zip ..."
unzip -q -n annotations_trainval2017.zip -d "$COCO_DIR"

# ---- Clean up zips (optional — saves ~20 GB) -----------------------------
echo ">>> Removing zip files to save space..."
rm -f "$COCO_DIR/train2017.zip" "$COCO_DIR/val2017.zip" "$COCO_DIR/annotations_trainval2017.zip"

# ---- Verify --------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "Verification:"
echo "  train2017 images : $(ls $COCO_DIR/train2017 | wc -l)  (expect 118,287)"
echo "  val2017 images   : $(ls $COCO_DIR/val2017   | wc -l)  (expect 5,000)"
echo "  annotations      : $(ls $COCO_DIR/annotations/*.json)"
echo "  Total size       : $(du -sh $COCO_DIR)"
echo ""
echo "Done: $(date)"
