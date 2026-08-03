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
