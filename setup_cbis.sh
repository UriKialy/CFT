#!/usr/bin/env bash
# =============================================================================
# setup_baselines_cbis.sh — one-time bootstrap to prepare GPS / SSF / SPT /
# PRO-VPT for running on cbis_ddsm. Assumes:
#   /workspace/{GPS,SSF,SPT,PRO-VPT}     freshly cloned upstream repos
#   /workspace/CFT                       this CFT repo (already patched)
#   /workspace/data/vtab-1k/cbis_ddsm    prepped VTAB-format data
#   /workspace/checkpoints/ViT-B_16.npz  ImageNet-21k checkpoint
#   /workspace/cft_integration           extracted CFT-SPT_SSF_run zip
#     (must contain external_artifacts/ and patches/)
#
# What it does:
#   1) Creates train800val200.txt symlink (VTAB class in SSF reads that name)
#   2) Copies modified upstream files from cft_integration into each repo
#   3) Adds 'cbis_ddsm' to every per-repo task registry / num_classes map
#   4) Symlinks ViT-B_16.npz into SPT/SSF/PRO-VPT/GPS checkpoints dirs
# =============================================================================
set -e

WORK=/workspace
DATA=$WORK/data/vtab-1k/cbis_ddsm
CKPT=$WORK/checkpoints/ViT-B_16.npz
INT=/workspace/CFT-spt-ssf

# --- 0) sanity
for d in "$WORK/GPS" "$WORK/SSF" "$WORK/SPT" "$WORK/PRO-VPT" "$DATA" "$INT"; do
    [ -e "$d" ] || { echo "MISSING: $d"; exit 1; }
done
[ -e "$CKPT" ] || { echo "MISSING: $CKPT (run wget for ViT-B_16.npz)"; exit 1; }

echo "==> 1) train800val200.txt symlink (SSF's VTAB class expects this name)"
ln -sf "$DATA/train800.txt" "$DATA/train800val200.txt"

echo "==> 2) Copy patches into each repo"
# SSF — data/{vtab.py,loader.py,dataset_factory.py,transforms_factory.py,__init__.py}, train.py, utils/
SSF_SRC=$INT/external_artifacts/upstream_modified_a100/SSF
SSF_DST=$WORK/SSF
mkdir -p $SSF_DST/data $SSF_DST/utils
cp -f $SSF_SRC/data/*.py $SSF_DST/data/
cp -f $SSF_SRC/utils/*.py $SSF_DST/utils/ 2>/dev/null || true
cp -f $INT/patches/SSF/train.py $SSF_DST/train.py

# GPS — train_gps.py + utils/pruning.py
cp -f $INT/external_artifacts/upstream_modified/GPS/train_gps.py $WORK/GPS/train_gps.py
mkdir -p $WORK/GPS/utils
cp -f $INT/patches/GPS/utils/pruning.py $WORK/GPS/utils/pruning.py 2>/dev/null || \
    cp -f $INT/patches/GPS/pruning.py $WORK/GPS/utils/pruning.py

# SPT — train_spt.py + model/ + datasets.py
SPT_SRC=$INT/external_artifacts/upstream_modified/SPT
SPT_DST=$WORK/SPT
mkdir -p $SPT_DST/model $SPT_DST/lib
cp -f $SPT_SRC/train_spt.py     $SPT_DST/train_spt.py
cp -rf $SPT_SRC/model/*         $SPT_DST/model/ 2>/dev/null || true
cp -rf $SPT_SRC/lib/*           $SPT_DST/lib/   2>/dev/null || true
cp -f $INT/patches/SPT/datasets.py $SPT_DST/datasets.py 2>/dev/null || true

# PRO-VPT — train.py + tune_vtab.py + src/data/loader.py + src/engine/trainer.py
PRO_SRC=$INT/external_artifacts/upstream_modified_a100/PRO-VPT
PRO_DST=$WORK/PRO-VPT
cp -f $PRO_SRC/train.py       $PRO_DST/train.py
cp -f $PRO_SRC/tune_vtab.py   $PRO_DST/tune_vtab.py 2>/dev/null || true
cp -f $INT/patches/PRO-VPT/src/data/loader.py     $PRO_DST/src/data/loader.py
cp -f $INT/patches/PRO-VPT/src/engine/trainer.py  $PRO_DST/src/engine/trainer.py
cp -f $INT/patches/PRO-VPT/src/models/vit_backbones/vit.py $PRO_DST/src/models/vit_backbones/vit.py 2>/dev/null || true

echo "==> 3) Add cbis_ddsm to each repo's task registry"
python3 - <<'PY'
from pathlib import Path
W = Path("/workspace")

def add_to_list(file, list_name, entry):
    """Insert entry into a Python list literal '<list_name> = [ ... ]' if not present."""
    if not file.exists():
        print(f"  [SKIP] {file} not found"); return
    s = file.read_text()
    if f"'{entry}'" in s or f'"{entry}"' in s:
        print(f"  [SKIP] {entry} already in {file.name}"); return
    # naive but works: find list start, insert before closing bracket
    import re
    m = re.search(rf"{list_name}\s*=\s*\[", s)
    if not m:
        print(f"  [WARN] {list_name} not found in {file.name}"); return
    # find matching ]
    i = m.end(); depth = 1
    while i < len(s) and depth:
        if s[i] == "[": depth += 1
        elif s[i] == "]": depth -= 1
        i += 1
    s = s[:i-1].rstrip() + f", '{entry}']" + s[i:]
    file.write_text(s)
    print(f"  [OK] added {entry} to {file.name}::{list_name}")

# SSF: _VTAB_DATASET in data/dataset_factory.py
add_to_list(W/"SSF/data/dataset_factory.py", "_VTAB_DATASET", "cbis_ddsm")

# GPS: usually has a similar list — try common names
for fname in ["train_gps.py", "utils/pruning.py"]:
    p = W/f"GPS/{fname}"
    if not p.exists(): continue
    s = p.read_text()
    for list_name in ["_VTAB_DATASET", "VTAB_DATASETS", "DATASETS"]:
        if list_name in s:
            add_to_list(p, list_name, "cbis_ddsm"); break

# SPT: datasets.py typically has dataset_dict / num_classes map
spt_ds = W/"SPT/datasets.py"
if spt_ds.exists():
    s = spt_ds.read_text()
    if "'cbis_ddsm'" not in s:
        # Look for the num_classes dict, append cbis_ddsm: 2
        import re
        m = re.search(r"(NUM_CLASSES|num_classes_map|class_dict)\s*=\s*\{", s)
        if m:
            i = m.end()
            # find balanced close
            depth = 1
            while i < len(s) and depth:
                if s[i] == "{": depth += 1
                elif s[i] == "}": depth -= 1
                i += 1
            s = s[:i-1].rstrip().rstrip(",") + ",\n    'cbis_ddsm': 2,\n}" + s[i:]
            spt_ds.write_text(s)
            print(f"  [OK] added cbis_ddsm: 2 to SPT/datasets.py")
        else:
            print(f"  [WARN] no num_classes map found in SPT/datasets.py — manual edit may be needed")

print("\n  (PRO-VPT uses YAML configs — see step 4 below.)")
PY

echo "==> 4) PRO-VPT YAML config for cbis_ddsm"
PROC=$WORK/PRO-VPT/configs/vtab/cbis_ddsm.yaml
mkdir -p "$(dirname $PROC)"
[ -f "$PROC" ] || cat > "$PROC" <<'YAML'
NUM_GPUS: 1
NUM_SHARDS: 1
OUTPUT_DIR: ""
RUN_N_TIMES: 1
MODEL:
  TRANSFER_TYPE: "prompt"
  TYPE: "vit"
  LINEAR:
    MLP_SIZES: []
  PROMPT:
    NUM_TOKENS: 5
    DEEP: True
DATA:
  NAME: "cbis_ddsm"
  DATAPATH: "/workspace/data/vtab-1k/cbis_ddsm"
  NUMBER_CLASSES: 2
  NUM_WORKERS: 4
  BATCH_SIZE: 32
SOLVER:
  TOTAL_EPOCH: 5
  WARMUP_EPOCH: 1
  BASE_LR: 0.001
  WEIGHT_DECAY: 0.0001
  OPTIMIZER: "adamw"
YAML
echo "  PRO-VPT yaml: $PROC"

echo "==> 5) Checkpoint symlinks"
for d in SPT SSF GPS PRO-VPT; do
    mkdir -p $WORK/$d/checkpoints
    ln -sf $CKPT $WORK/$d/checkpoints/ViT-B_16.npz
done

echo "==> DONE."
