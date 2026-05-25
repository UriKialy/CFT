#!/bin/bash
# =============================================================================
# VTAB-1K Dataset Setup
#
# The VTAB-1K benchmark uses 1000 training samples (800 train + 200 val)
# from 19 visual tasks. The standard split files come from the SSF repo.
#
# Usage:
#   bash setup_data.sh                    # download to ./data/vtab-1k
#   bash setup_data.sh /path/to/data      # download to custom location
# =============================================================================

set -e

DATA_ROOT="${1:-./data}"
VTAB_DIR="$DATA_ROOT/vtab-1k"

echo "VTAB-1K setup — target: $VTAB_DIR"

# -- Option 1: From the SSF repo (most common source) --
# The SSF paper repo hosts VTAB-1K splits ready to use.
# Clone it and copy the dataset:
if [ ! -d "$VTAB_DIR" ]; then
    echo ""
    echo "Downloading VTAB-1K from the SSF repo..."
    echo ""

    TMP_DIR=$(mktemp -d)
    git clone --depth 1 https://github.com/dongzelian/SSF.git "$TMP_DIR/SSF"

    if [ -d "$TMP_DIR/SSF/datasets/vtab-1k" ]; then
        mkdir -p "$DATA_ROOT"
        cp -r "$TMP_DIR/SSF/datasets/vtab-1k" "$VTAB_DIR"
        echo "Copied dataset to $VTAB_DIR"
    else
        echo "ERROR: vtab-1k not found in SSF repo."
        echo ""
        echo "The SSF repo may use Git LFS for the dataset."
        echo "Try manually:"
        echo "  1. git lfs install"
        echo "  2. git clone https://github.com/dongzelian/SSF.git"
        echo "  3. cp -r SSF/datasets/vtab-1k $VTAB_DIR"
        echo ""
        echo "Alternative: download from Google Drive / OneDrive if you"
        echo "have a vtab-1k.tar file, then extract it:"
        echo "  mkdir -p $VTAB_DIR && tar xf vtab-1k.tar -C $DATA_ROOT"
        rm -rf "$TMP_DIR"
        exit 1
    fi

    rm -rf "$TMP_DIR"
else
    echo "Already exists: $VTAB_DIR"
fi

# -- Verify --
echo ""
echo "Checking tasks..."
EXPECTED_TASKS="caltech101 cifar clevr_count clevr_dist dmlab dsprites_loc dsprites_ori dtd diabetic_retinopathy eurosat kitti oxford_flowers102 oxford_iiit_pet patch_camelyon resisc45 smallnorb_azi smallnorb_ele sun397 svhn"
FOUND=0
MISSING=0

for task in $EXPECTED_TASKS; do
    if [ -d "$VTAB_DIR/$task" ]; then
        FOUND=$((FOUND + 1))
    else
        echo "  MISSING: $task"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "Found: $FOUND/19 tasks"
if [ $MISSING -gt 0 ]; then
    echo "WARNING: $MISSING tasks missing!"
else
    echo "All 19 VTAB-1K tasks present."
fi

echo ""
echo "Done. Update config.py data_dir to: $VTAB_DIR"
