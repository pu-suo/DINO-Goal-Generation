#!/usr/bin/env bash
# Phase-0 environment on a vast.ai GPU box (single RTX 4090).
#
# Built to PERSIST across vast.ai Stop/Start reboots: conda AND the dino_wm env
# live under /workspace (the ONLY path vast.ai keeps; /venv, /root/.bashrc and apt
# packages are reset to the base image on reboot). Re-running this is cheap and
# idempotent -- after a reboot it just reinstalls the few apt libs and re-points
# conda; the Python env itself is reused from /workspace.
#
# We deliberately SKIP mujoco / mujoco-py / d4rl (only point_maze needs them; the
# pointmaze import is optional in env/__init__.py). DINOv2 is fetched at runtime
# via torch.hub. Python is pinned to 3.10 because hydra-core 1.2.0 crashes on 3.11+.
set -e

WS=${WS:-/workspace}
CONDA_DIR="$WS/miniconda3"
ENV_PREFIX="$WS/envs/dino_wm"
PYV=3.10

echo "==> system libs (wiped on reboot; cheap to reinstall). Non-fatal if they fail."
apt-get update -y && apt-get install -y --no-install-recommends \
  unzip wget git ffmpeg libgl1 libglib2.0-0 || true

echo "==> conda under $WS (survives Stop/Start)"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "==> env $ENV_PREFIX (python $PYV)"
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=$PYV
fi
conda activate "$ENV_PREFIX"
pip install --upgrade pip

echo "==> GPU torch — pinned to cu121 (repo's stack; the default 'pip install torch'"
echo "    grabs a cu128 wheel that needs a newer driver than most vast.ai boxes have)"
python -c "import torch" 2>/dev/null \
  || pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

echo "==> PushT / DINO-WM runtime deps (OLD gym API — never install gymnasium)"
pip install "numpy<2" "gym==0.23.1" "pymunk==6.8.0" "pygame==2.5.2" \
  shapely opencv-python-headless scikit-image einops "hydra-core==1.2.0" "omegaconf==2.3.0" \
  "hydra-submitit-launcher==1.2.0" \
  decord imageio imageio-ffmpeg matplotlib transformers accelerate wandb submitit psutil scikit-learn
# opencv-python-headless (not opencv-python): headless boxes lack libGL. cv2 ENCODES
# the dataset mp4s in-process (gen_pusht_multicolor) -- robust where ffmpeg deadlocks.

echo "==> writing $WS/activate.sh (source this in every shell)"
cat > "$WS/activate.sh" <<EOF
# Activate the dino_wm env + project env vars. Source after every login/reboot:
#   source $WS/activate.sh
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export DATASET_DIR=$WS/data
export CKPTS=$WS/ckpts
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=disabled
EOF

# auto-source for the current container lifetime (note: ~/.bashrc itself is reset on reboot)
grep -q "workspace/activate.sh" ~/.bashrc 2>/dev/null || echo "source $WS/activate.sh" >> ~/.bashrc

echo ""
echo "==> DONE."
echo "    Now:                    source $WS/activate.sh"
echo "    Get data (once):        bash scripts/download_data.sh"
echo "    After a Stop/Start:     bash scripts/setup_vastai.sh   # fast: reuses the /workspace env"
echo "                            source $WS/activate.sh"
