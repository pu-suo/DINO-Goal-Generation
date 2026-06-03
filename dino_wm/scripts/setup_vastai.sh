#!/usr/bin/env bash
# Phase-0 environment on a vast.ai GPU box (single RTX 4090).
#
# Creates conda env `dino_wm` with exactly the deps PushT + DINO-WM need. We
# deliberately SKIP mujoco / mujoco-py / d4rl (from the upstream environment.yaml):
# only point_maze needs them, the pointmaze import is optional in env/__init__.py,
# and mujoco-py is the main source of setup failures. DINOv2 is fetched at runtime
# via torch.hub. This is the validated stack (same as the local dev env).
set -e

ENVNAME=${ENVNAME:-dino_wm}
PYV=3.10

echo "==> system libs (headless rendering + video)"
apt-get update -y && apt-get install -y --no-install-recommends \
  unzip wget git ffmpeg libgl1 libglib2.0-0 || true

echo "==> conda env $ENVNAME (python $PYV)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENVNAME" python=$PYV
conda activate "$ENVNAME"
pip install --upgrade pip

echo "==> GPU torch — pinned to cu121 (repo's stack; the default 'pip install torch'"
echo "    grabs a cu128 wheel that needs a newer driver than most vast.ai boxes have)"
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

echo "==> PushT / DINO-WM runtime deps (OLD gym API — never install gymnasium)"
pip install "numpy<2" "gym==0.23.1" "pymunk==6.8.0" "pygame==2.5.2" \
  shapely opencv-python-headless scikit-image einops "hydra-core==1.2.0" "omegaconf==2.3.0" \
  "hydra-submitit-launcher==1.2.0" \
  decord imageio imageio-ffmpeg matplotlib transformers accelerate wandb submitit psutil scikit-learn
# NOTE: opencv-python-headless (not opencv-python): headless boxes lack libGL, and
# the non-headless wheel fails to import there. cv2 is used to ENCODE the dataset
# mp4s in-process (gen_pusht_multicolor) -- robust where ffmpeg/libx264 deadlocks.

echo ""
echo "==> Done. In your shell run:   conda activate $ENVNAME"
echo "    Then: export SDL_VIDEODRIVER=dummy   (headless pygame)"
echo "    Get data with: bash scripts/download_data.sh   (after setting DATASET_DIR + CKPTS)"
