"""Dump the G1 re-validation corpus on the NEW (dot) render.

Per episode captures: START (clean + dot), a few ROLLOUT dots (dot near contact), GOAL (clean).
Saves both clean (effector-free) and dot (EE-dot) renders per frame + EE pos + per-block state
+ the source tuple (start_block, target_block). Enables:
  - NEW clean baseline (clean frames) -- re-measured on the corrected render (NOT old 0.92/96)
  - Slice B (dot frames, dot patches maskable via EE)
  - goal-pairs displacement check (start/goal states + tuple)

Run (langtable env):
  /Users/Tom/miniforge3/envs/langtable/bin/python dino_wm/scripts/langtable/lt_dump_g1.py --episodes 40 --seed 0 --out /tmp/lt_g1.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lt_compat  # noqa: E402
lt_compat.install_tf_agents_shim()

from language_table.environments import blocks  # noqa: E402
from language_table.environments import language_table  # noqa: E402
from language_table.environments.oracles import push_oracle_rrt_slowdown  # noqa: E402
from language_table.environments.rewards import block2block  # noqa: E402
import lt_render  # noqa: E402

ORDER = list(blocks.FIXED_8_COMBINATION)
SIZE = 224
ROLLOUT_STEPS = (10, 25, 40)  # capture dot frames here during the push (near-contact)


def record(env, ee_xy):
    state = env.compute_state()
    clean = lt_render.render_topdown(env, SIZE, mode="clean")
    dot = lt_render.render_topdown(env, SIZE, mode="dot", ee_xy=ee_xy)
    xy = np.array([np.asarray(state[f"block_{b}_translation"]).ravel()[:2] for b in ORDER], np.float32)
    yaw = np.array([np.asarray(state[f"block_{b}_orientation"]).ravel()[0] for b in ORDER], np.float32)
    mask = np.array([np.asarray(state[f"block_{b}_mask"]).ravel()[0] for b in ORDER], np.float32)
    return clean, dot, xy, yaw, mask


def ee(aenv):
    return np.asarray(aenv.last_obs["effector_translation"]).ravel()[:2].astype(np.float32)


def decode_instr(aenv):
    instr = aenv.last_obs.get("instruction")
    return bytes([c for c in np.asarray(instr).ravel().tolist() if c]).decode("utf-8", "ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", default="/tmp/lt_g1.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward, control_frequency=10.0, seed=args.seed)
    aenv = lt_compat.GymToTFAgentsEnv(env)
    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(aenv, use_ee_planner=True)

    cols = {k: [] for k in ["clean", "dot", "ee", "block_xy", "block_yaw", "block_mask",
                            "kind", "episode", "instruction", "start_block", "target_block"]}
    n_succ = 0

    def add(c, d, xy, yaw, m, e, kind, ep, instr, sb, tb):
        cols["clean"].append(c); cols["dot"].append(d); cols["ee"].append(e)
        cols["block_xy"].append(xy); cols["block_yaw"].append(yaw); cols["block_mask"].append(m)
        cols["kind"].append(kind); cols["episode"].append(ep); cols["instruction"].append(instr)
        cols["start_block"].append(sb); cols["target_block"].append(tb)

    for e in range(args.episodes):
        ts = None
        for _ in range(50):
            ts = aenv.reset(); oracle.reset()
            if oracle.get_plan(aenv.compute_state()):
                break
        rc = env._reward_calculator
        sb = getattr(rc, "_start_block", ""); tb = getattr(rc, "_target_block", "")
        instr = decode_instr(aenv)
        eestart = ee(aenv)
        add(*record(env, eestart), eestart, "start", e, instr, sb, tb)
        step = 0
        while not ts.is_last() and step < 200:
            ts = aenv.step(oracle.action(ts, ()).action); step += 1
            if step in ROLLOUT_STEPS:
                er = ee(aenv)
                add(*record(env, er), er, "rollout", e, instr, sb, tb)
        if aenv.succeeded:
            n_succ += 1
            eg = ee(aenv)
            add(*record(env, eg), eg, "goal", e, instr, sb, tb)
        if (e + 1) % 10 == 0:
            print(f"  {e+1}/{args.episodes} eps, {n_succ} success, {len(cols['clean'])} frames")

    out = {k: np.stack(v) if k in ("clean", "dot", "ee", "block_xy", "block_yaw", "block_mask")
           else np.array(v) for k, v in cols.items()}
    out.update(blocks=np.array(ORDER), half_extent=np.float32(lt_render.DEFAULT_HALF_EXTENT),
               center=np.array(lt_render.CENTER, np.float32), size=np.int32(SIZE),
               cam_z=np.float32(lt_render.DEFAULT_CAM_Z))
    np.savez_compressed(args.out, **out)
    print(f"SAVED {args.out}: {len(cols['clean'])} frames ({n_succ}/{args.episodes} goals) "
          f"kinds={dict(zip(*np.unique(out['kind'], return_counts=True)))}")


if __name__ == "__main__":
    main()
