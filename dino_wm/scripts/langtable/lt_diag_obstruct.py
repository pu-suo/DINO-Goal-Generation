"""Lever C.1 (langtable env): for each eval seed, is the straight A->B carrot path obstructed by
another block? Replays the eval's exact reset (env.seed(s) + valid-init loop, matching lt_envserver),
computes min clearance of other blocks to the A->B segment. Prints per-seed; join offline with the
baseline successes to test whether FAILURES are enriched for obstruction (=> Lever C is justified)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lt_compat  # noqa
lt_compat.install_tf_agents_shim()
from language_table.environments import blocks  # noqa
from language_table.environments import language_table  # noqa
from language_table.environments.oracles import push_oracle_rrt_slowdown  # noqa
from language_table.environments.rewards import block2block  # noqa
ORDER = list(blocks.FIXED_8_COMBINATION)


def seg_dist(p, a, b):
    ab = b - a; denom = float(ab @ ab)
    t = 0.0 if denom < 1e-9 else float(np.clip((p - a) @ ab / denom, 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clearance", type=float, default=0.05)   # block needs this gap from the A->B line
    a = ap.parse_args()
    env = language_table.LanguageTable(block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
                                       reward_factory=block2block.BlockToBlockReward,
                                       control_frequency=10.0, seed=a.seed)
    aenv = lt_compat.GymToTFAgentsEnv(env)
    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(aenv, use_ee_planner=True)

    def xy():
        st = env.compute_state()
        return np.array([np.asarray(st[f"block_{b}_translation"]).ravel()[:2] for b in ORDER], np.float32)

    print("seed start_block target_block d0 n_obstruct min_clear")
    n_obstr = 0
    for s in range(a.n):
        try:
            env.seed(a.seed + s)
        except Exception:
            pass
        ts = None
        for _ in range(50):
            ts = aenv.reset(); oracle.reset()
            try:
                if oracle.get_plan(aenv.compute_state()):
                    break
            except Exception:
                continue
        rc = env._reward_calculator
        ai = ORDER.index(str(rc._start_block)); bi = ORDER.index(str(rc._target_block))
        p = xy(); A, B = p[ai], p[bi]; d0 = float(np.linalg.norm(A - B))
        clears = [seg_dist(p[k], A, B) for k in range(8) if k not in (ai, bi)]
        nob = int(sum(c < a.clearance for c in clears)); mc = float(min(clears))
        n_obstr += int(nob > 0)
        print(f"{a.seed+s} {ORDER[ai]} {ORDER[bi]} {d0:.3f} {nob} {mc:.3f}")
    print(f"# obstructed (>=1 block within {a.clearance}u of A->B segment): {n_obstr}/{a.n}")


if __name__ == "__main__":
    main()
