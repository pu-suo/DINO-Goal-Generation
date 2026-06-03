"""
Phase 0.1 unit tests for the multi-color PushT env.

Run from the dino_wm/ directory with the dev env, e.g.:
    cd dino_wm
    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python -m pytest tests/test_multicolor_env.py -q
or just:
    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python tests/test_multicolor_env.py
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

import numpy as np

from env.pusht.pusht_env import PushTEnv
from env.pusht.pusht_multicolor_env import PushTMultiColorEnv
from env.pusht.multicolor_common import tee_coverage, get_palette
from env.pusht import multicolor_sampler as mcs


# --- decorrelation ------------------------------------------------------------
def test_named_target_decorrelated_from_nearest():
    rate, chance, n = mcs.nearest_target_predicts_named(n_samples=4000, n_targets=4, seed=1)
    assert abs(rate - chance) < 0.03, f"named==nearest rate {rate:.3f} vs chance {chance:.3f}"


def test_named_target_position_uncorrelated_with_block():
    bx, tx = [], []
    for i in range(3000):
        lay = mcs.sample_layout(7_000 + i, n_targets=4)
        bx.append(lay["init_state"][2])
        tx.append(lay["goal_pose"][0])
    corr = np.corrcoef(bx, tx)[0, 1]
    assert abs(corr) < 0.06, f"block_x vs named_target_x corr too high: {corr:.3f}"


# --- visual-only: targets must not perturb physics ----------------------------
def test_targets_are_visual_only():
    init = np.array([200, 300, 256, 256, 0.3, 0.0, 0.0], dtype=np.float64)
    actions = np.array([[0.2, -0.1], [-0.1, 0.3], [0.25, 0.25], [0.0, -0.3]])

    base = PushTEnv(with_velocity=True, with_target=True)
    base.seed(0); base.reset_to_state = init.copy(); base.reset()
    base_states = []
    for a in actions:
        _, _, _, info = base.step(a)
        base_states.append(info["state"])

    mc = PushTMultiColorEnv(with_velocity=True, n_targets=4)
    mc.sample_and_set_layout(123)        # arbitrary decals
    mc.seed(0); mc.reset_to_state = init.copy(); mc.reset()
    mc_states = []
    for a in actions:
        _, _, _, info = mc.step(a)
        mc_states.append(info["state"])

    base_states, mc_states = np.array(base_states), np.array(mc_states)
    # block pose (idx 2:5) must be identical -> decals have no physical effect
    assert np.allclose(base_states[:, 2:5], mc_states[:, 2:5], atol=1e-6), \
        f"block trajectory diverged:\n{base_states[:, 2:5]}\n{mc_states[:, 2:5]}"


# --- start frame is identical across instructions (text load-bearing) ---------
def test_frame_invariant_to_active_target():
    env = PushTMultiColorEnv(with_velocity=True, n_targets=4, render_size=224)
    layout = mcs.sample_layout(55, n_targets=4)
    init = layout["init_state"]

    frames = []
    for active in (0, len(layout["targets"]) - 1):
        layout2 = dict(layout)
        layout2["active_idx"] = active
        layout2["goal_pose"] = layout["targets"][active]["pose"].copy()
        env.set_layout(layout2)
        env.seed(0); env.reset_to_state = init.copy()
        obs, _ = env.reset()
        frames.append(obs["visual"].copy())
    assert np.array_equal(frames[0], frames[1]), \
        "start frame changed with the named target -> info leaked into pixels"


def test_all_target_colors_render():
    env = PushTMultiColorEnv(with_velocity=True, n_targets=4, render_size=224, outline_thickness=8)
    env.sample_and_set_layout(9)
    obs, _ = env.reset()
    img = obs["visual"].reshape(-1, 3).astype(np.int32)
    for name, rgb in get_palette(4):
        d = np.abs(img - np.array(rgb)).max(axis=1)
        assert (d < 60).sum() >= 1, f"color {name} {rgb} not visible in rendered frame"


# --- named-target coverage success -------------------------------------------
def test_eval_state_named_target():
    env = PushTMultiColorEnv(with_velocity=True, n_targets=4, success_threshold=0.95)
    layout = mcs.sample_layout(3, n_targets=4)
    env.set_layout(layout)
    gp = layout["goal_pose"]
    goal_state = np.array([100, 100, gp[0], gp[1], gp[2], 0, 0], dtype=np.float64)

    # block exactly on the named target -> coverage 1, success
    res = env.eval_state(goal_state, goal_state.copy())
    assert res["coverage"] > 0.999 and res["success"]
    # block far away -> failure
    far = goal_state.copy(); far[2] += 200; far[3] += 200
    res2 = env.eval_state(goal_state, far)
    assert res2["coverage"] < 0.05 and not res2["success"]


def test_tee_coverage_self_is_one():
    p = (256.0, 256.0, 0.7)
    assert abs(tee_coverage(p, p) - 1.0) < 1e-6


# --- split scaffold -----------------------------------------------------------
def test_combo_split_clean():
    train, test = mcs.make_combo_split(n_targets=4, n_bins=3, heldout_frac=0.2, seed=0)
    assert train.isdisjoint(test), "train/test combos overlap (leakage)"
    assert len(test) > 0
    # every color and every bin still present in train (recombination, not exclusion)
    names = {c for c, _ in get_palette(4)}
    assert {c for c, _ in train} == names
    assert {b for _, b in train} == set(range(9))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
