"""Slice A: synthesize same-color / same-shape CONTACT configs (boundary-to-boundary), clean.

Places a chosen pair of blocks in contact (centers at rA+rB), the other 6 spread, all
effector-free, and renders the clean top-down frame + per-block state. Categories:
  same-color: red(moon,pentagon) blue(cube,moon) green(cube,star) yellow(pentagon,star)
  same-shape: moon(red,blue) pentagon(red,yellow) cube(blue,green) star(green,yellow)

Run (langtable env):
  .../langtable/bin/python dino_wm/scripts/langtable/lt_dump_contact.py --per_cat 30 --out /tmp/lt_contact.npz
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lt_compat  # noqa: E402
lt_compat.install_tf_agents_shim()

from language_table.environments import blocks  # noqa: E402
from language_table.environments import language_table  # noqa: E402
from language_table.environments.rewards import block2block  # noqa: E402
import lt_render  # noqa: E402

ORDER = list(blocks.FIXED_8_COMBINATION)
IDX = {b: i for i, b in enumerate(ORDER)}
SIZE = 224
X_MIN, X_MAX, Y_MIN, Y_MAX, BUF = 0.15, 0.6, -0.3048, 0.3048, 0.05

CATEGORIES = {
    "sc_red": ("red_moon", "red_pentagon"), "sc_blue": ("blue_cube", "blue_moon"),
    "sc_green": ("green_cube", "green_star"), "sc_yellow": ("yellow_pentagon", "yellow_star"),
    "ss_moon": ("red_moon", "blue_moon"), "ss_pentagon": ("red_pentagon", "yellow_pentagon"),
    "ss_cube": ("blue_cube", "green_cube"), "ss_star": ("green_star", "yellow_star"),
}


def quat(client, yaw):
    return client.getQuaternionFromEuler([math.pi / 2, 0, yaw])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cat", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/lt_contact.npz")
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward, control_frequency=10.0, seed=args.seed)
    aenv = lt_compat.GymToTFAgentsEnv(env)
    aenv.reset()
    client = env.pybullet_client
    bid = {b: env._block_to_pybullet_id[b] for b in ORDER}
    # resting z + per-block radius from current (settled) poses
    rest_z = client.getBasePositionAndOrientation(bid[ORDER[0]])[0][2]
    radius = {}
    for b in ORDER:
        lo, hi = client.getAABB(bid[b])
        radius[b] = float(((hi[0] - lo[0]) + (hi[1] - lo[1])) / 4.0)
    print(f"rest_z={rest_z:.4f} radii={ {k: round(v,3) for k,v in radius.items()} }")

    def place(b, x, y):
        client.resetBasePositionAndOrientation(
            bid[b], [float(x), float(y), rest_z], quat(client, rng.uniform(0, 2 * math.pi)))

    def rand_xy():
        return rng.uniform([X_MIN + BUF, Y_MIN + BUF], [X_MAX - BUF, Y_MAX - BUF])

    cols = {k: [] for k in ["clean", "block_xy", "block_yaw", "block_mask", "category",
                            "pair_a", "pair_b", "episode"]}
    cfg = 0
    for cat, (A, B) in CATEGORIES.items():
        for _ in range(args.per_cat):
            placed = []  # (x,y)
            # pair A at random; B at contact distance in a random dir (in-bounds)
            ax, ay = rand_xy()
            # span overlapping-merged (0.04, hardest) -> boundary-touch (rA+rB), covering
            # the block2block goal regime (centers < 0.05 success radius).
            d = rng.uniform(0.04, radius[A] + radius[B])
            for _try in range(40):
                th = rng.uniform(0, 2 * math.pi)
                bx, by = ax + d * math.cos(th), ay + d * math.sin(th)
                if X_MIN + BUF <= bx <= X_MAX - BUF and Y_MIN + BUF <= by <= Y_MAX - BUF:
                    break
            place(A, ax, ay); place(B, bx, by)
            placed = [(ax, ay), (bx, by)]
            # other 6 spread (min 0.085 apart from all placed)
            for b in ORDER:
                if b in (A, B):
                    continue
                for _try in range(60):
                    x, y = rand_xy()
                    if all(math.hypot(x - px, y - py) > 0.085 for px, py in placed):
                        break
                place(b, x, y); placed.append((x, y))
            state = env.compute_state()
            clean = lt_render.render_topdown(env, SIZE, mode="clean")
            xy = np.array([np.asarray(state[f"block_{b}_translation"]).ravel()[:2] for b in ORDER], np.float32)
            yaw = np.array([np.asarray(state[f"block_{b}_orientation"]).ravel()[0] for b in ORDER], np.float32)
            m = np.array([np.asarray(state[f"block_{b}_mask"]).ravel()[0] for b in ORDER], np.float32)
            cols["clean"].append(clean); cols["block_xy"].append(xy)
            cols["block_yaw"].append(yaw); cols["block_mask"].append(m)
            cols["category"].append(cat); cols["pair_a"].append(IDX[A])
            cols["pair_b"].append(IDX[B]); cols["episode"].append(cfg)
            cfg += 1

    out = {k: (np.stack(v) if k in ("clean", "block_xy", "block_yaw", "block_mask") else np.array(v))
           for k, v in cols.items()}
    out.update(blocks=np.array(ORDER), half_extent=np.float32(lt_render.DEFAULT_HALF_EXTENT),
               center=np.array(lt_render.CENTER, np.float32), size=np.int32(SIZE))
    np.savez_compressed(args.out, **out)
    print(f"SAVED {args.out}: {cfg} configs, {args.per_cat}/category, contact dist=rA+rB")


if __name__ == "__main__":
    main()
