"""Full-trajectory corpus writer for G2 (DINO-WM dynamics + CEM).

Oracle rollouts; per env-step logs: dot-render RGB (224, pusher=dot), 2D action, EE xy
(proprio), 26-dim state. Per trajectory also saves the effector-free CLEAN goal render +
goal state (the oracle's reached end-state = reachable-by-construction goal for the G2
plannability smoke). One npz per trajectory in --out_dir.

Run (langtable env):
  .../langtable/bin/python dino_wm/scripts/langtable/lt_dump_traj.py --episodes 20 --seed 0 --out_dir /workspace/lt_traj
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


def state_vec(state):
    xy = np.array([np.asarray(state[f"block_{b}_translation"]).ravel()[:2] for b in ORDER], np.float32)
    yaw = np.array([np.asarray(state[f"block_{b}_orientation"]).ravel()[0] for b in ORDER], np.float32)
    mask = np.array([np.asarray(state[f"block_{b}_mask"]).ravel()[0] for b in ORDER], np.float32)
    return xy, yaw, mask


def ee(aenv):
    return np.asarray(aenv.last_obs["effector_translation"]).ravel()[:2].astype(np.float32)


def decode_instr(aenv):
    instr = aenv.last_obs.get("instruction")
    return bytes([c for c in np.asarray(instr).ravel().tolist() if c]).decode("utf-8", "ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/workspace/lt_traj")
    ap.add_argument("--max_steps", type=int, default=120)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward, control_frequency=10.0, seed=args.seed)
    aenv = lt_compat.GymToTFAgentsEnv(env)
    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(aenv, use_ee_planner=True)

    n_saved = 0
    for e in range(args.episodes):
        ts = None
        for _ in range(50):
            ts = aenv.reset(); oracle.reset()
            if oracle.get_plan(aenv.compute_state()):
                break
        rc = env._reward_calculator
        sb = str(getattr(rc, "_start_block", "")); tb = str(getattr(rc, "_target_block", ""))
        instr = decode_instr(aenv)
        frames, acts, props, sxy, syaw, smask = [], [], [], [], [], []
        step = 0
        while not ts.is_last() and step < args.max_steps:
            st = env.compute_state()
            eexy = ee(aenv)
            frames.append(lt_render.render_topdown(env, SIZE, mode="dot", ee_xy=eexy))
            xy, yaw, m = state_vec(st)
            props.append(eexy); sxy.append(xy); syaw.append(yaw); smask.append(m)
            a = oracle.action(ts, ()).action.astype(np.float32)
            acts.append(a)
            ts = aenv.step(a); step += 1
        # final frame (no action)
        st = env.compute_state(); eexy = ee(aenv)
        frames.append(lt_render.render_topdown(env, SIZE, mode="dot", ee_xy=eexy))
        xy, yaw, m = state_vec(st); props.append(eexy); sxy.append(xy); syaw.append(yaw); smask.append(m)
        acts.append(np.zeros(2, np.float32))
        # effector-free goal (reached end-state) + goal state
        goal_clean = lt_render.render_topdown(env, SIZE, mode="clean")
        goal_xy, goal_yaw, goal_mask = state_vec(st)
        succ = bool(aenv.succeeded)
        if succ:
            n_saved += 1
        np.savez_compressed(
            os.path.join(args.out_dir, f"w{args.seed}_t{e:04d}.npz"),
            frames=np.stack(frames), actions=np.stack(acts), proprio=np.stack(props),
            block_xy=np.stack(sxy), block_yaw=np.stack(syaw), block_mask=np.stack(smask),
            goal_clean=goal_clean, goal_xy=goal_xy, goal_yaw=goal_yaw, goal_mask=goal_mask,
            instruction=instr, start_block=sb, target_block=tb, success=succ,
            seq_length=len(frames), blocks=np.array(ORDER),
            half_extent=np.float32(lt_render.DEFAULT_HALF_EXTENT),
            center=np.array(lt_render.CENTER, np.float32), size=np.int32(SIZE))
        if (e + 1) % 5 == 0:
            print(f"  {e+1}/{args.episodes} eps, {n_saved} success, last T={len(frames)}")
    print(f"DONE seed={args.seed}: {args.episodes} trajs ({n_saved} success) -> {args.out_dir}")


if __name__ == "__main__":
    main()
