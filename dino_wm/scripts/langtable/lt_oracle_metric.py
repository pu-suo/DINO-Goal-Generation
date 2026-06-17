"""Lever B.2 validation (langtable env): run the scripted oracle and record the TRUE final ||A-B||
(from env.compute_state at the real episode end, NOT the frameskip-subsampled cache). Reports oracle
success at several thresholds + the contact-distance distribution. This is the anti-gaming gate: the
metric we adopt must score the genuinely-completing oracle highly, and the threshold must be geometry-
justified, never tuned to the eval number.
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

ORDER = list(blocks.FIXED_8_COMBINATION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=200)
    a = ap.parse_args()
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward, control_frequency=10.0, seed=a.seed)
    aenv = lt_compat.GymToTFAgentsEnv(env)
    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(aenv, use_ee_planner=True)

    def AB(ai, bi):
        st = env.compute_state()
        pa = np.asarray(st[f"block_{ORDER[ai]}_translation"]).ravel()[:2]
        pb = np.asarray(st[f"block_{ORDER[bi]}_translation"]).ravel()[:2]
        return float(np.linalg.norm(pa - pb))

    finals, succs = [], []
    for e in range(a.episodes):
        ts = None
        for _ in range(50):
            ts = aenv.reset(); oracle.reset()
            try:
                if oracle.get_plan(aenv.compute_state()):
                    break
            except Exception:  # noqa: BLE001
                continue
        rc = env._reward_calculator
        ai = ORDER.index(str(rc._start_block)); bi = ORDER.index(str(rc._target_block))
        step = 0
        while not ts.is_last() and step < a.max_steps:
            ts = aenv.step(oracle.action(ts, ()).action); step += 1
        finals.append(AB(ai, bi)); succs.append(bool(aenv.succeeded))
    finals = np.array(finals); succs = np.array(succs)
    print(f"ORACLE n={len(finals)} seed={a.seed}")
    print(f"  env-success (aenv.succeeded) rate = {succs.mean():.3f}")
    print(f"  TRUE final ||A-B|| (compute_state at real end): mean={finals.mean():.4f} "
          f"median={np.median(finals):.4f} p90={np.percentile(finals,90):.4f} max={finals.max():.4f}")
    for thr in [0.04, 0.05, 0.06, 0.07, 0.08]:
        print(f"  oracle success @ final ||A-B|| < {thr:.2f}: {np.mean(finals<thr):.3f}")
    print("VALIDATION: the threshold where oracle>=0.95 (if any) is the loosest geometry-plausible")
    print("relational metric. If even 0.08 stays <0.95, the oracle simply fails ~that fraction (not a")
    print("metric problem) and 0.05u (the env's block2block def) stands as ground truth.")


if __name__ == "__main__":
    main()
