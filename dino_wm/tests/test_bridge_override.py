"""CPU tests for the Stage-2 bridge goal override (planning/cem.py z_obs_g_override).

The override must (a) default to None = stock behavior (goal encoded from obs_g),
(b) when set, replace the goal latent the objective sees WITHOUT encoding obs_g,
(c) propagate through the fast-config chunked path (traj_chunk>1) unchanged.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planning.cem import CEMPlanner

B, T, P, D, A, H = 2, 1, 4, 3, 2, 2  # evals, time, patches, dim, action_dim, horizon


class FakeWM(torch.nn.Module):
    """encode_obs stamps the obs's fill value into the latent so tests can tell
    WHICH observation a latent came from; rollouts return zeros."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.encode_calls = []

    def encode_obs(self, obs):
        val = float(obs["visual"].flatten()[0])
        self.encode_calls.append(val)
        b = obs["visual"].shape[0]
        return {"visual": torch.full((b, T, P, D), val),
                "proprio": torch.zeros(b, T, 2)}

    def rollout_from_zobs(self, z_obs_0, act):
        n = act.shape[0]
        return ({"visual": torch.zeros(n, H + 1, P, D),
                 "proprio": torch.zeros(n, H + 1, 2)}, None)

    def rollout(self, obs_0, act):
        n = act.shape[0]
        return ({"visual": torch.zeros(n, H + 1, P, D),
                 "proprio": torch.zeros(n, H + 1, 2)}, None)


class IdPrep:
    def transform_obs(self, obs):
        return obs


class SpyObjective:
    def __init__(self):
        self.seen_goal_vals = []

    def __call__(self, z_pred, z_goal, vis_mask=None):
        self.seen_goal_vals.append(float(z_goal["visual"].flatten()[0]))
        return torch.arange(z_pred["visual"].shape[0], dtype=torch.float32)


class NullWandb:
    def log(self, *a, **k):
        pass


def make_planner(obj, **kw):
    return CEMPlanner(horizon=H, topk=2, num_samples=4, var_scale=1.0, opt_steps=1,
                      eval_every=999, wm=FakeWM(), action_dim=A, objective_fn=obj,
                      preprocessor=IdPrep(), evaluator=None, wandb_run=NullWandb(),
                      log_filename=None, **kw)


def obs(val):
    return {"visual": torch.full((B, T, 3, 8, 8), float(val)),
            "proprio": torch.zeros(B, T, 2)}


def override(val):
    return {"visual": torch.full((B, T, P, D), float(val)),
            "proprio": torch.zeros(B, T, 2)}


def test_default_encodes_goal():
    obj = SpyObjective()
    p = make_planner(obj)
    p.plan(obs_0=obs(1.0), obs_g=obs(2.0))
    assert obj.seen_goal_vals and all(v == 2.0 for v in obj.seen_goal_vals)
    assert 2.0 in p.wm.encode_calls  # obs_g WAS encoded


def test_override_replaces_goal_without_encoding():
    obj = SpyObjective()
    p = make_planner(obj)
    p.z_obs_g_override = override(42.0)
    p.plan(obs_0=obs(1.0), obs_g=obs(2.0))
    assert obj.seen_goal_vals and all(v == 42.0 for v in obj.seen_goal_vals)
    assert 2.0 not in p.wm.encode_calls  # obs_g never encoded


def test_override_through_chunked_fast_path():
    obj = SpyObjective()
    p = make_planner(obj, traj_chunk=2, skip_succeeded=False)
    p.z_obs_g_override = override(7.0)
    p.plan(obs_0=obs(1.0), obs_g=obs(2.0))
    assert obj.seen_goal_vals and all(v == 7.0 for v in obj.seen_goal_vals)
    assert 2.0 not in p.wm.encode_calls


if __name__ == "__main__":
    test_default_encodes_goal()
    test_override_replaces_goal_without_encoding()
    test_override_through_chunked_fast_path()
    print("ALL OK")
