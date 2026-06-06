"""Dataset for QRL quasimetric training over cached pusht_noise model-step latents.

Reads the cache produced by scripts/cache_qm_latents.py and yields, per item, the
two sample kinds QRL needs:

  * a TRANSITION pair (z_t, z_{t+1}) -- consecutive model-steps in one trajectory,
    used for the local-cost constraint   relu(d(z_t,z_{t+1}) + r)^2 <= eps^2  (r=-1).
  * a VALUE pair (z_s, z_g)            -- z_s a random model-step, z_g a future state
    in the same trajectory (optionally a random cross-trajectory state), used for
    the spreading objective  max E[phi(d(z_s, z_g))].

MASK CONSISTENCY (the load-bearing detail): for each pair we drop the UNION of the
two pushers' patches from BOTH latents, using the exact planner-side helper
(`manipulator_energy_mask`). So d_theta always sees two grids carrying the same
keep-mask -- in training and at planning time alike. `keep` is returned per pair.
"""
import json
from pathlib import Path

import numpy as np
import torch

from env.pusht.multicolor_common import manipulator_energy_mask


class QMLatentDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, split="train", mask_dilation=0,
                 p_random_goal=0.0, max_goal_offset=None, seed=0):
        """
        cache_dir: <data_path>/qm_latents (parent of the per-split folders).
        p_random_goal: fraction of VALUE pairs whose goal is a random cross-traj
                       state instead of a same-traj future state (QRL spreading).
        max_goal_offset: cap on same-traj future offset in model-steps (None = to end).
        """
        d = Path(cache_dir) / split
        self.latents = torch.load(d / "latents.pth")            # (Ntot,196,384) f16
        self.states = torch.load(d / "states.pth").float()      # (Ntot,state_dim)
        self.starts = torch.load(d / "traj_starts.pth").tolist()
        self.lengths = torch.load(d / "traj_lengths.pth").tolist()
        with open(d / "meta.json") as f:
            self.meta = json.load(f)
        self.pusher_xy = self.states[:, 0:2].numpy().astype(np.float64)  # sim coords
        self.mask_dilation = int(mask_dilation)
        self.p_random_goal = float(p_random_goal)
        self.max_goal_offset = max_goal_offset
        self.Ntot = self.latents.shape[0]

        # enumerate within-traj transitions (i, i+1) and remember each global index's
        # (traj_id, pos_in_traj) so we can sample a same-traj future goal.
        self.trans = []
        self.traj_of = np.empty(self.Ntot, dtype=np.int64)
        self.pos_of = np.empty(self.Ntot, dtype=np.int64)
        for tid, (s, L) in enumerate(zip(self.starts, self.lengths)):
            for p in range(L):
                self.traj_of[s + p] = tid
                self.pos_of[s + p] = p
            for p in range(L - 1):
                self.trans.append((s + p, s + p + 1))
        self.trans = np.asarray(self.trans, dtype=np.int64)        # (Ntrans,2)
        self._mask_cache = {}
        self.rng = np.random.RandomState(seed)
        print(f"[QMLatentDataset:{split}] {self.Ntot} model-steps, {len(self.trans)} "
              f"transitions, {len(self.lengths)} trajs; mask_dilation={mask_dilation}, "
              f"p_random_goal={p_random_goal}")

    def __len__(self):
        return len(self.trans)

    def _keep(self, gi, gj):
        """Union keep-mask (P,) for the two model-steps gi, gj (rounded-xy cached)."""
        a = self.pusher_xy[gi]; b = self.pusher_xy[gj]
        key = (round(float(a[0])), round(float(a[1])),
               round(float(b[0])), round(float(b[1])), self.mask_dilation)
        m = self._mask_cache.get(key)
        if m is None:
            m = manipulator_energy_mask([a, b], dilation=self.mask_dilation)  # (196,) f32
            self._mask_cache[key] = m
        return torch.from_numpy(m)

    def _sample_goal(self, gs):
        """Pick a goal global-index for value-pair anchored at global index gs."""
        if self.rng.rand() < self.p_random_goal:
            return int(self.rng.randint(self.Ntot))             # random cross-traj state
        tid = int(self.traj_of[gs]); s = self.starts[tid]; L = self.lengths[tid]
        pos = int(self.pos_of[gs])
        if pos >= L - 1:
            return gs                                           # already last -> self (d~0)
        hi = L - 1
        if self.max_goal_offset is not None:
            hi = min(hi, pos + int(self.max_goal_offset))
        gpos = int(self.rng.randint(pos + 1, hi + 1))
        return s + gpos

    def __getitem__(self, idx):
        gi, gj = self.trans[idx]                                # transition pair
        gs = int(self.rng.randint(self.Ntot))                   # value-pair anchor (random state)
        gg = self._sample_goal(gs)                              # value-pair goal
        return {
            "z_a": self.latents[gi].float(), "z_b": self.latents[gj].float(),
            "keep_ab": self._keep(gi, gj),
            "z_s": self.latents[gs].float(), "z_g": self.latents[gg].float(),
            "keep_sg": self._keep(gs, gg),
        }


def trajectory_views(cache_dir, split):
    """Helper for the validator: return (latents f32, states, list-of (start,len),
    pusher_xy, meta) so monotonicity can be checked along whole trajectories."""
    d = Path(cache_dir) / split
    latents = torch.load(d / "latents.pth").float()
    states = torch.load(d / "states.pth").float()
    starts = torch.load(d / "traj_starts.pth").tolist()
    lengths = torch.load(d / "traj_lengths.pth").tolist()
    with open(d / "meta.json") as f:
        meta = json.load(f)
    return latents, states, list(zip(starts, lengths)), states[:, 0:2].numpy().astype(np.float64), meta
