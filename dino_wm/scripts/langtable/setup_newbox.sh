#!/usr/bin/env bash
# Provision a fresh vast.ai box (image: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime) for
# Language-Table gen + train. Base /opt/conda already has torch 2.3+cu121 + torchvision.
# Adds: the repo (models.vit + langtable kit), einops, and the `langtable` conda env.
# Replicates the dino4090 layout so run_stage1.sh + the kit run unchanged. Idempotent.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
REPO=https://github.com/pu-suo/DINO-Goal-Generation.git

echo "=== [1/5] apt: git ==="
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }

echo "=== [2/5] clone repo -> /workspace/dino_goal ==="
mkdir -p /workspace/envs
if [ -d /workspace/dino_goal/.git ]; then git -C /workspace/dino_goal pull -q; else git clone --depth 1 "$REPO" /workspace/dino_goal; fi
ln -sfn /workspace/dino_goal/dino_wm/scripts/langtable /workspace/langtable_kit

echo "=== [3/5] dino_wm env = base pytorch (/opt/conda) + einops ==="
ln -sfn /opt/conda /workspace/envs/dino_wm
/workspace/envs/dino_wm/bin/python -m pip install -q einops
/workspace/envs/dino_wm/bin/python -c "import torch,torchvision,einops,numpy; print('dino_wm OK: torch',torch.__version__,'cuda',torch.cuda.is_available())"

echo "=== [4/5] langtable env (setup_langtable.sh; uses /opt/conda) ==="
bash /workspace/langtable_kit/setup_langtable.sh

echo "=== [5/5] verify langtable ==="
/workspace/envs/langtable/bin/python -c "from language_table.environments import blocks; print('langtable OK:', len(blocks.FIXED_8_COMBINATION), 'blocks')"
echo "=== NEWBOX READY ==="
