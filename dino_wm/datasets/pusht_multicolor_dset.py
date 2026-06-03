"""
Phase 0.2 dataloaders for the multi-color PushT dataset.

Two views over the same on-disk dataset (see scripts/gen_pusht_multicolor.py):

1. `PushTMultiColorDataset` / `load_pusht_multicolor_slice_train_val`
   DINO-WM-compatible trajectory view -> (obs, act, state, env_info), where
   `env_info` is a *layout dict* (env.update_env(env_info) re-installs the N
   decals) plus the multi-color label fields. Drop-in for plan.py / train.py.

2. `PushTMultiColorLatentGoalDataset` / `load_multicolor_latent_goal`
   The bridge (`g`) view -> (z_start, instruction, z_goal, label), built from the
   cached DINOv2 latents (scripts/cache_latents.py). This is the Phase-0.2 gate
   deliverable; `g` is not trained until Phase 1.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset
from env.pusht.multicolor_common import DEFAULT_PALETTE

_RGB_OF = dict(DEFAULT_PALETTE)
ACTION_SCALE = 100.0


# --- label <-> layout ---------------------------------------------------------
def label_to_layout(label):
    """Reconstruct an env layout dict (with rgb) from a stored per-episode label."""
    poses = np.asarray(label["target_poses"])
    bins = label.get("target_bins", [-1] * len(label["target_colors"]))
    targets = [
        {"color": c, "rgb": _RGB_OF[c], "pose": np.asarray(poses[i], dtype=np.float64), "bin": int(bins[i])}
        for i, c in enumerate(label["target_colors"])
    ]
    return {
        "shape": "T",  # upstream env_info compatibility
        "targets": targets,
        "active_idx": int(label["active_idx"]),
        "active_color": label["active_color"],
        "goal_pose": np.asarray(label["goal_pose"], dtype=np.float64),
        "init_state": np.asarray(label["init_state"], dtype=np.float64),
        "instruction": label["instruction"],
        "template_id": int(label["template_id"]),
    }


def _read_video_frames(vid_path, frames):
    """THWC uint8 tensor for the given frame indices (decord, imageio fallback)."""
    frames = list(frames)
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(vid_path), num_threads=1)
        return vr.get_batch(frames)
    except Exception:
        import imageio.v2 as imageio
        rd = imageio.get_reader(str(vid_path))
        return torch.from_numpy(np.stack([rd.get_data(i) for i in frames]))


def _load_stats(data_path, state_dim, proprio_dim, action_dim):
    """Load dataset stats (saved at the dataset root); identity if missing."""
    stats_file = Path(data_path).parent / "stats.pth"
    if stats_file.exists():
        s = torch.load(stats_file)
        return (s["action_mean"][:action_dim].float(), s["action_std"][:action_dim].float(),
                s["state_mean"][:state_dim].float(), s["state_std"][:state_dim].float(),
                s["proprio_mean"][:proprio_dim].float(), s["proprio_std"][:proprio_dim].float())
    z = lambda d: (torch.zeros(d), torch.ones(d))
    return (*z(action_dim), *z(state_dim), *z(proprio_dim))


class PushTMultiColorDataset(TrajDataset):
    def __init__(self, data_path, transform=None, n_rollout=None,
                 normalize_action=True, with_velocity=True):
        self.data_path = Path(data_path)
        self.transform = transform
        self.with_velocity = with_velocity

        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "rel_actions.pth").float() / ACTION_SCALE
        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        with open(self.data_path / "labels.pkl", "rb") as f:
            self.labels = pickle.load(f)
        shapes_file = self.data_path / "shapes.pkl"
        self.shapes = pickle.load(open(shapes_file, "rb")) if shapes_file.exists() else ["T"] * len(self.states)

        n = n_rollout or len(self.states)
        self.states, self.actions = self.states[:n], self.actions[:n]
        self.seq_lengths, self.labels = self.seq_lengths[:n], self.labels[:n]
        self.proprios = self.states[..., :2].clone()
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth")[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            (self.action_mean, self.action_std, self.state_mean, self.state_std,
             self.proprio_mean, self.proprio_std) = _load_stats(
                self.data_path, self.state_dim, self.proprio_dim, self.action_dim)
        else:
            self.action_mean, self.action_std = torch.zeros(self.action_dim), torch.ones(self.action_dim)
            self.state_mean, self.state_std = torch.zeros(self.state_dim), torch.ones(self.state_dim)
            self.proprio_mean, self.proprio_std = torch.zeros(self.proprio_dim), torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
        print(f"Loaded {n} multi-color rollouts from {self.data_path}")

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat([self.actions[i, :self.seq_lengths[i]] for i in range(len(self.seq_lengths))], dim=0)

    def get_frames(self, idx, frames):
        image = _read_video_frames(self.data_path / "obses" / f"episode_{idx:06d}.mp4", frames)
        image = rearrange(image / 255.0, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": self.proprios[idx, frames]}
        env_info = label_to_layout(self.labels[idx])
        return obs, self.actions[idx, frames], self.states[idx, frames], env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_pusht_multicolor_slice_train_val(
    transform, n_rollout=None, data_path="data/pusht_multicolor",
    normalize_action=True, split_ratio=0.9, num_hist=0, num_pred=0,
    frameskip=0, with_velocity=True, **kwargs,
):
    """Returns ({train,valid} sliced) , ({train,valid} traj) -- mirrors pusht loader.

    Uses the pre-generated train/ and val/ folders (the dynamics split). The
    held-out color-location 'test/' split is loaded separately for eval.
    """
    train_dset = PushTMultiColorDataset(
        data_path=data_path + "/train", transform=transform,
        n_rollout=n_rollout, normalize_action=normalize_action, with_velocity=with_velocity)
    val_dset = PushTMultiColorDataset(
        data_path=data_path + "/val", transform=transform,
        n_rollout=n_rollout, normalize_action=normalize_action, with_velocity=with_velocity)

    num_frames = num_hist + num_pred
    datasets = {"train": TrajSlicerDataset(train_dset, num_frames, frameskip),
                "valid": TrajSlicerDataset(val_dset, num_frames, frameskip)}
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset


# --- bridge (g) view: (z_start, instruction, z_goal, label) -------------------
class PushTMultiColorLatentGoalDataset(torch.utils.data.Dataset):
    """Cached-latent view for training/evaluating the bridge `g`.

    Reads <latent_dir>/<split>/{start_latents.pth, goal_latents.pth} (each
    (N,196,384)) produced by scripts/cache_latents.py, plus the split labels.
    """
    def __init__(self, latent_dir, data_path, split="train"):
        self.start = torch.load(Path(latent_dir) / split / "start_latents.pth")
        self.goal = torch.load(Path(latent_dir) / split / "goal_latents.pth")
        with open(Path(data_path) / split / "labels.pkl", "rb") as f:
            self.labels = pickle.load(f)
        assert len(self.start) == len(self.goal) == len(self.labels), "latent/label count mismatch"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        return {
            "z_start": self.start[idx],            # (196, 384)
            "z_goal": self.goal[idx],              # (196, 384)
            "instruction": label["instruction"],
            "active_color": label["active_color"],
            "active_idx": int(label["active_idx"]),
            "goal_pose": torch.as_tensor(np.asarray(label["goal_pose"]), dtype=torch.float32),
            "label": label,
        }


def load_multicolor_latent_goal(latent_dir, data_path, splits=("train", "val", "test")):
    return {sp: PushTMultiColorLatentGoalDataset(latent_dir, data_path, sp) for sp in splits}
