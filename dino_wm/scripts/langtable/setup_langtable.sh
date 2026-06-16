#!/usr/bin/env bash
# Reproducible minimal Language Table install (env + oracle + render; NO TF/JAX/reverb).
# Idempotent. Tested target: Linux vast.ai box (headless) and macOS arm64.
#
# Usage:
#   bash setup_langtable.sh [LT_DIR] [ENV_PREFIX] [PATCH_FILE]
# Defaults (box): LT_DIR=/workspace/language-table  ENV_PREFIX=/workspace/envs/langtable
set -euo pipefail

LT_DIR="${1:-/workspace/language-table}"
ENV_PREFIX="${2:-/workspace/envs/langtable}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${3:-$HERE/langtable_minimal.patch}"

echo "=== [1/4] clone language-table -> $LT_DIR ==="
if [ ! -d "$LT_DIR/.git" ]; then
  git clone --depth 1 https://github.com/google-research/language-table.git "$LT_DIR"
else
  echo "  already cloned"
fi

echo "=== [2/4] apply minimal-install patch (gfile/tf shims) ==="
if [ -f "$PATCH_FILE" ]; then
  if git -C "$LT_DIR" apply --check "$PATCH_FILE" 2>/dev/null; then
    git -C "$LT_DIR" apply "$PATCH_FILE" && echo "  patch applied"
  else
    echo "  patch already applied or not applicable (skipping)"
  fi
else
  echo "  WARN: patch file not found at $PATCH_FILE — env import will fail on tensorflow"
fi

echo "=== [3/4] conda env @ $ENV_PREFIX (py3.10) ==="
# locate conda
for c in /workspace/miniconda3 "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
  [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
done
[ -d "$ENV_PREFIX" ] || conda create -y -p "$ENV_PREFIX" python=3.10
PY="$ENV_PREFIX/bin/python"
"$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade

echo "=== [4/4] pip minimal deps (no TF/JAX/reverb/tf-agents) ==="
"$PY" -m pip install --upgrade pip setuptools wheel
# opencv-python-headless: no libGL needed on headless boxes.
"$PY" -m pip install "numpy==1.23.5" scipy "gym==0.23.0" pybullet opencv-python-headless \
  matplotlib absl-py six protobuf imageio
"$PY" -m pip install --no-deps -e "$LT_DIR"

echo "=== verify import (no tensorflow) ==="
"$ENV_PREFIX/bin/python" - <<'PY'
from language_table.environments import language_table, blocks
from language_table.environments.rewards import block2block
print("language_table import OK; BLOCK_8 =", len(blocks.FIXED_8_COMBINATION), "blocks")
PY
echo "=== DONE. interpreter: $ENV_PREFIX/bin/python ==="
