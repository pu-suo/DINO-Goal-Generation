"""Coordinate (clean-scene) bridge view: (z_start, z_goal, spec) for training/eval `g`.

Reads <latent_dir>/<split>/{start_latents.pth, goal_latents.pth, specs.pth,
start_poses.pth} produced by scripts/cache_coord_latents.py over the single-T clean
dataset from scripts/gen_pusht_coord.py.

The `spec` is the goal BLOCK pose (x, y, theta) in sim-512 coords -- the coordinate
front-end (models/bridge.py CoordSpecEncoder, Phase 2) Fourier-features (x/512, y/512)
and encodes theta as (sin, cos) plus a soft 2D Gaussian heatmap over the 14x14 patch
grid. The loader stays representation-agnostic and just serves the raw pose; nothing
here is normalized so the front-end owns the parametrization.
"""
from pathlib import Path

import torch


class PushTCoordLatentGoalDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dir, data_path=None, split="train"):
        d = Path(latent_dir) / split
        self.start = torch.load(d / "start_latents.pth")      # (N,196,384)
        self.goal = torch.load(d / "goal_latents.pth")        # (N,196,384)
        self.spec = torch.load(d / "specs.pth").float()       # (N,3) = goal (x,y,theta) sim-512
        self.start_pose = torch.load(d / "start_poses.pth").float()  # (N,3) sim-512
        assert len(self.start) == len(self.goal) == len(self.spec) == len(self.start_pose), \
            "latent/spec count mismatch"

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, idx):
        return {
            "z_start": self.start[idx],          # (196, 384)
            "z_goal": self.goal[idx],            # (196, 384)
            "spec": self.spec[idx],              # (3,) goal (x,y,theta) sim-512
            "start_pose": self.start_pose[idx],  # (3,) start block (x,y,theta) sim-512
        }


def load_coord_latent_goal(latent_dir, data_path=None, splits=("train", "val", "test")):
    return {sp: PushTCoordLatentGoalDataset(latent_dir, data_path, sp) for sp in splits}
