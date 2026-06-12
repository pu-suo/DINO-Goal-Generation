"""StrideOneLatentDataset must reproduce TrajSlicerDataset's windows exactly (modulo the
visual being pre-encoded latents instead of images). We check the action/proprio/state
PAIRINGS and the visual FRAME INDICES match window-for-window on synthetic data, so the
stride-1 cached recipe == the original recipe — the whole point of the per-frame test.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.traj_dset import TrajSlicerDataset, TrajDataset
from datasets.dyn_latent_dset import StrideOneLatentDataset

T, NF, FS, NTRAJ = 30, 4, 5, 3
PDIM, ADIM, SDIM, EMB, P = 4, 2, 5, 8, 6  # small


class ToyTraj(TrajDataset):
    """Each frame's visual is a constant grid = its global frame id, so a latent's value
    reveals exactly which env-frame produced it -> lets us assert frame-index pairing."""
    def __init__(self):
        self.actions = torch.arange(NTRAJ * T * ADIM, dtype=torch.float32).reshape(NTRAJ, T, ADIM)
        self.proprios = torch.arange(NTRAJ * T * PDIM, dtype=torch.float32).reshape(NTRAJ, T, PDIM)
        self.states = torch.arange(NTRAJ * T * SDIM, dtype=torch.float32).reshape(NTRAJ, T, SDIM)
        self.gid = torch.arange(NTRAJ * T, dtype=torch.float32).reshape(NTRAJ, T)  # frame id
        self.action_dim, self.proprio_dim, self.state_dim = ADIM, PDIM, SDIM

    def get_seq_length(self, i): return T
    def __len__(self): return NTRAJ

    def get_frames(self, i, frames):
        fr = list(frames)
        vis = self.gid[i, fr][:, None, None, None].repeat(1, 3, P, P)  # (len,3,P,P) = frame id
        return {"visual": vis, "proprio": self.proprios[i, fr]}, self.actions[i, fr], self.states[i, fr], {}

    def __getitem__(self, i): return self.get_frames(i, range(T))


def build_pf_cache(tmp):
    """Write a per-frame dyn_latents_pf cache whose 'visual' latent encodes the frame id."""
    d = ToyTraj()
    visual = d.gid[:, :, None, None].repeat(1, 1, P * P, EMB).half()      # (N,T,P*P,EMB), value=frame id
    import json
    from pathlib import Path
    out = Path(tmp) / "train"; out.mkdir(parents=True)
    torch.save(visual, out / "visual.pth")
    torch.save(d.proprios, out / "proprio.pth")
    torch.save(d.actions, out / "actions.pth")
    torch.save(d.states, out / "states.pth")
    json.dump({"frameskip": FS, "cache_stride": 1, "S": T, "n_traj": NTRAJ,
               "proprio_dim": PDIM, "action_dim": ADIM}, open(out / "meta.json", "w"))
    return out


def test_stride_one_matches_trajslicer():
    import tempfile
    tmp = tempfile.mkdtemp()
    pf = build_pf_cache(tmp)

    orig = TrajSlicerDataset(ToyTraj(), NF, FS)          # the reference recipe
    cached = StrideOneLatentDataset(pf, NF)

    # same window count + same (traj,start) slice set (both enumerate start in range(T-NF*FS+1))
    assert len(orig) == len(cached), (len(orig), len(cached))

    # TrajSlicerDataset permutes slices; index by (i,start) instead. Rebuild orig unpermuted.
    orig.slices = sorted([tuple(int(x) for x in s) for s in orig.slices])
    cached.slices = np.array(sorted([tuple(int(x) for x in s) for s in cached.slices]))

    for k in range(len(orig)):
        o_obs, o_act, o_state = orig[k]
        c_obs, c_act, c_state = cached[k]
        # action/proprio/state pairings identical
        assert torch.equal(o_act, c_act), f"action mismatch @ {k}"
        assert torch.equal(o_obs["proprio"], c_obs["proprio"]), f"proprio mismatch @ {k}"
        assert torch.equal(o_state, c_state), f"state mismatch @ {k}"
        # visual FRAME INDICES: orig visual is (NF,3,P,P) const=frame id; cached latent (NF,P*P,EMB) const=frame id
        o_fid = o_obs["visual"][:, 0, 0, 0]
        c_fid = c_obs["visual"][:, 0, 0]
        assert torch.equal(o_fid, c_fid), f"visual frame-index mismatch @ {k}: {o_fid} vs {c_fid}"
    print(f"OK: {len(orig)} windows match TrajSlicerDataset (actions, proprio, state, visual frame indices)")


if __name__ == "__main__":
    test_stride_one_matches_trajslicer()
    print("ALL OK")
