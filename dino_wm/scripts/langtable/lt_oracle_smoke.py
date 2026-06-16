"""G0 oracle smoke: run the scripted RRT oracle on block2block, no TF/tf-agents.

Run with the `langtable` conda env:
  /Users/Tom/miniforge3/envs/langtable/bin/python dino_wm/scripts/langtable/lt_oracle_smoke.py [--episodes N]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lt_compat  # noqa: E402

# Install the tf_agents shim BEFORE importing the oracle (which imports tf_agents).
lt_compat.install_tf_agents_shim()

from language_table.environments import blocks  # noqa: E402
from language_table.environments import language_table  # noqa: E402
from language_table.environments.oracles import push_oracle_rrt_slowdown  # noqa: E402
from language_table.environments.rewards import block2block  # noqa: E402


def decode_instruction(obs):
    instr = obs.get("instruction")
    if instr is None:
        return None
    return bytes([c for c in np.asarray(instr).ravel().tolist() if c]).decode(
        "utf-8", "ignore")


def run_episode(aenv, oracle, max_steps=200, max_init_tries=50):
    # Find a valid init (oracle can motion-plan), per eval/main.py.
    ts = None
    for _ in range(max_init_tries):
        ts = aenv.reset()
        if hasattr(oracle, "reset"):
            oracle.reset()
        raw_state = aenv.compute_state()
        try:
            if oracle.get_plan(raw_state):
                break
        except Exception as e:  # noqa: BLE001
            print(f"  get_plan raised {type(e).__name__}: {e}")
    instr = decode_instruction(aenv.last_obs)
    steps = 0
    while not ts.is_last() and steps < max_steps:
        pstep = oracle.action(ts, ())
        ts = aenv.step(pstep.action)
        steps += 1
    return aenv.succeeded, steps, instr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0,
        seed=args.seed,
    )
    aenv = lt_compat.GymToTFAgentsEnv(env)
    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
        aenv, use_ee_planner=True)

    n_succ = 0
    for ep in range(args.episodes):
        succ, steps, instr = run_episode(aenv, oracle)
        n_succ += int(bool(succ))
        print(f"ep {ep}: success={bool(succ)}  steps={steps}  instr={instr!r}")
    print(f"ORACLE_SMOKE: {n_succ}/{args.episodes} succeeded")


if __name__ == "__main__":
    main()
