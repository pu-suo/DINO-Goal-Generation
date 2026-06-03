#!/usr/bin/env bash
# Download + arrange the PushT dataset + shipped checkpoint from OSF (Phase 0).
# Requires DATASET_DIR and CKPTS to be set to paths on a disk with ~12 GB free.
set -e

: "${DATASET_DIR:?set DATASET_DIR, e.g. export DATASET_DIR=/workspace/data}"
: "${CKPTS:?set CKPTS, e.g. export CKPTS=/workspace/ckpts}"
VO="a56a296ce3b24cceaf408383a175ce28"   # OSF view-only token
mkdir -p "$DATASET_DIR" "$CKPTS"
command -v unzip >/dev/null || { apt-get update -y && apt-get install -y unzip; }

echo "==> pusht_noise.zip (2.8 GB)"
wget -c -O "$DATASET_DIR/pusht_noise.zip" "https://osf.io/download/k2d8w/?view_only=$VO"
echo "==> outputs.zip (953 MB — all checkpoints incl. pusht)"
wget -c -O "$CKPTS/outputs.zip" "https://osf.io/download/xvzs4/?view_only=$VO"

echo "==> unzipping"
unzip -q -o "$DATASET_DIR/pusht_noise.zip" -d "$DATASET_DIR"
unzip -q -o "$CKPTS/outputs.zip" -d "$CKPTS"

echo "==> verifying the exact paths the code reads"
ok=1
ls "$DATASET_DIR"/pusht_noise/train/states.pth >/dev/null 2>&1 && ls "$DATASET_DIR"/pusht_noise/val/states.pth >/dev/null 2>&1 \
  && echo "  [ok] dataset: $DATASET_DIR/pusht_noise/{train,val}" \
  || { echo "  [!!] dataset structure unexpected — run: find $DATASET_DIR/pusht_noise -maxdepth 2 -type d"; ok=0; }
ls "$CKPTS"/outputs/pusht/hydra.yaml >/dev/null 2>&1 && ls "$CKPTS"/outputs/pusht/checkpoints/model_latest.pth >/dev/null 2>&1 \
  && echo "  [ok] checkpoint: $CKPTS/outputs/pusht" \
  || { echo "  [!!] checkpoint structure unexpected — run: find $CKPTS -maxdepth 3"; ok=0; }

[ "$ok" = 1 ] && echo "==> ALL GOOD. (you may rm the .zip files to save space)" \
             || echo "==> Fix the paths above (mv the contents), then re-verify."
