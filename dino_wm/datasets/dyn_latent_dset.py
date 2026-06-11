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
    def __init__(self, dyn_dir, num_frames):
        d = Path(dyn_dir)
        self.meta = json.load(open(d / "meta.json"))
        self.fs = self.meta["frameskip"]
        self.S = self.meta["S"]
        self.num_frames = num_frames
        self.visual = torch.load(d / "visual.pth")      # (n, S, 196, 384) fp16
        self.proprio = torch.load(d / "proprio.pth")    # (n, S, pdim) fp32
        self.actions = torch.load(d / "actions.pth")    # (n, T, adim) fp32
        self.states = torch.load(d / "states.pth")      # (n, S, sdim) fp32
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
