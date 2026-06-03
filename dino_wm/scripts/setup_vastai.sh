#!/usr/bin/env bash
# One-time environment setup on a vast.ai box (Ubuntu + NVIDIA, single RTX 4090).
# Creates the upstream `dino_wm` conda env and installs MuJoCo 210. Phase 0 needs
# no deps beyond environment.yaml. Run from the dino_wm/ directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # dino_wm/
cd "$HERE"

echo "==> conda env create (dino_wm) from environment.yaml"
if ! conda env list | grep -q '/dino_wm$'; then
  conda env create -f environment.yaml
else
  echo "    env 'dino_wm' already exists; skipping (use 'conda env update -f environment.yaml' to refresh)"
fi

echo "==> MuJoCo 210 (repo env setup expects it; PushT itself is pymunk)"
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  mkdir -p "$HOME/.mujoco"
  wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P "$HOME/.mujoco/"
  tar -xzf "$HOME/.mujoco/mujoco210-linux-x86_64.tar.gz" -C "$HOME/.mujoco"
fi
LINE='export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'"$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
grep -qF "$LINE" "$HOME/.bashrc" || echo "$LINE" >> "$HOME/.bashrc"

cat <<'EOF'

==> Done. Next:
    conda activate dino_wm
    source ~/.bashrc            # MuJoCo LD_LIBRARY_PATH
    export DATASET_DIR=/data    # must contain pusht_noise/{train,val}
    export CKPTS=/ckpts         # pusht checkpoint under $CKPTS/outputs/pusht

  Download from OSF (https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28):
    - dataset 'pusht_noise'        -> $DATASET_DIR/pusht_noise
    - checkpoint 'pusht'           -> $CKPTS/outputs/pusht
  Then follow specs/PHASE_0_RUNBOOK.md (start at 0.0).
EOF
