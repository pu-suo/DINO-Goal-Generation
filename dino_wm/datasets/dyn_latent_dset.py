"""Cached model-step latent dataset for fast dynamics training (see
scripts/cache_dynamics_latents.py). Yields windows of `num_frames` consecutive
model-steps -- the same (visual, proprio, action, state) tensors train.py's
TrajSlicerDataset produces for a GRID-ALIGNED start, but with the visual already
encoded so VWorldModel.forward_latent can skip the DINOv2 forward.

A window at model-step start s (env-frame start s*fs):
    visual : vis[i, s : s+num_frames]                       (num_frames, 196, 384)
    proprio: prop[i, s : s+num_frames]                      (num_frames, proprio_dim)
    act    : actions[i, s*fs : (s+num_frames)*fs] -> (num_frames, fs*action_dim)
             (concatenated fs env-actions per model-step, exactly as TrajSlicerDataset)
    state  : states[i, s : s+num_frames]                    (num_frames, state_dim)
"""
import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset


class DynLatentSliceDataset(Dataset):
    def __init__(self, dyn_dir, num_frames, max_traj=None):
        d = Path(dyn_dir)
        self.meta = json.load(open(d / "meta.json"))
        self.fs = self.meta["frameskip"]
        self.S = self.meta["S"]
        self.num_frames = num_frames
        self.visual = torch.load(d / "visual.pth")      # (n, S, 196, 384) fp16
        self.proprio = torch.load(d / "proprio.pth")    # (n, S, pdim) fp32
        self.actions = torch.load(d / "actions.pth")    # (n, T, adim) fp32
        self.states = torch.load(d / "states.pth")      # (n, S, sdim) fp32
        if max_traj is not None:                        # scaling-curve subset (first k trajs)
            self.visual, self.proprio = self.visual[:max_traj], self.proprio[:max_traj]
            self.actions, self.states = self.actions[:max_traj], self.states[:max_traj]
        self.action_dim = self.actions.shape[-1] * self.fs
        self.proprio_dim = self.proprio.shape[-1]
        self.state_dim = self.states.shape[-1]
        n = self.visual.shape[0]
        self.slices = [(i, s) for i in range(n) for s in range(self.S - num_frames + 1)]
        self.slices = np.array(self.slices)
        print(f"DynLatentSliceDataset: {n} trajs x {self.S} steps -> {len(self.slices)} windows")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, s = (int(x) for x in self.slices[idx])
        nf, fs = self.num_frames, self.fs
        visual = self.visual[i, s:s + nf].float()
        proprio = self.proprio[i, s:s + nf]
        act = self.actions[i, s * fs:(s + nf) * fs]
        act = rearrange(act, "(n f) d -> n (f d)", n=nf)
        state = self.states[i, s:s + nf]
        return {"visual": visual, "proprio": proprio}, act, state


class StrideOneLatentDataset(Dataset):
    """Per-frame cached-latent dataset that reproduces datasets/traj_dset.py
    TrajSlicerDataset EXACTLY (stride-1 window starts -> full phase augmentation, all
    fs sub-phases), to test data scaling under the ORIGINAL recipe without the
    grid-alignment confound. Requires cache_dynamics_latents.py --cache_stride 1
    (per-frame latents under dyn_latents_pf/). A window at env-frame start s:
        visual : visual[i, s : s+nf*fs : fs]   (nf frames, fs apart -- same as obs[s:end:fs])
        act    : actions[i, s : s+nf*fs] -> rearrange '(n f) d -> n (f d)'
        proprio: proprio[i, s : s+nf*fs : fs]
    For start = k*fs these windows coincide with DynLatentSliceDataset's; the extra
    fs-1 phases per start are the augmentation the grid cache drops.
    """
    def __init__(self, dyn_dir, num_frames, max_traj=None):
        d = Path(dyn_dir)
        self.meta = json.load(open(d / "meta.json"))
        assert self.meta.get("cache_stride") == 1, "StrideOneLatentDataset needs a per-frame cache (cache_stride=1)"
        self.fs = self.meta["frameskip"]
        self.num_frames = num_frames
        self.visual = torch.load(d / "visual.pth")      # (n, T, 196, 384) fp16
        self.proprio = torch.load(d / "proprio.pth")    # (n, T, pdim) fp32
        self.actions = torch.load(d / "actions.pth")    # (n, T, adim) fp32
        self.states = torch.load(d / "states.pth")      # (n, T, sdim) fp32
        if max_traj is not None:
            self.visual, self.proprio = self.visual[:max_traj], self.proprio[:max_traj]
            self.actions, self.states = self.actions[:max_traj], self.states[:max_traj]
        self.action_dim = self.actions.shape[-1] * self.fs
        self.proprio_dim = self.proprio.shape[-1]
        self.state_dim = self.states.shape[-1]
        n, T = self.visual.shape[0], self.visual.shape[1]
        win = num_frames * self.fs
        self.slices = np.array([(i, s) for i in range(n) for s in range(T - win + 1)])
        print(f"StrideOneLatentDataset: {n} trajs x {T} frames -> {len(self.slices)} windows "
              f"(stride-1, {T - win + 1}/traj)")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, s = (int(x) for x in self.slices[idx])
        nf, fs = self.num_frames, self.fs
        end = s + nf * fs
        visual = self.visual[i, s:end:fs].float()
        proprio = self.proprio[i, s:end:fs]
        act = rearrange(self.actions[i, s:end], "(n f) d -> n (f d)", n=nf)
        state = self.states[i, s:end:fs]
        return {"visual": visual, "proprio": proprio}, act, state
