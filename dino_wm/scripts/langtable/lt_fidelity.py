"""G1-prep render-vs-state fidelity: does the top-down render match the 26-dim state?

For each block, predict its pixel from compute_state() + world_to_pixel(), then find
the nearest same-color blob centroid in the (effector-hidden) render and measure the
pixel error. Aggregated over R resets. This is the plan's "decode pose from render,
match state" check that gates G1. Also verifies effector-hidden removes the pusher.

Run: /Users/Tom/miniforge3/envs/langtable/bin/python dino_wm/scripts/langtable/lt_fidelity.py [--resets R]
"""
import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lt_compat  # noqa: E402
lt_compat.install_tf_agents_shim()

import cv2  # noqa: E402
from language_table.environments import blocks  # noqa: E402
from language_table.environments import language_table  # noqa: E402
from language_table.environments.rewards import block2block  # noqa: E402
import lt_render  # noqa: E402

SIZE = 224
HALF = lt_render.DEFAULT_HALF_EXTENT
PX_PER_UNIT = SIZE / (2 * HALF)            # ~350 px / world-unit
MATCH_PX = 8.0                             # ~1 block radius
COLORS = {"red": (170, 40, 40), "green": (40, 150, 50),
          "blue": (45, 60, 175), "yellow": (215, 190, 55)}


def color_mask(img, color, thresh=85.0):
    d = np.linalg.norm(img.astype(np.float32) - np.array(color, np.float32), axis=2)
    return (d < thresh).astype(np.uint8)


def nearest_blob_dist(img, color_name, pred_col, pred_row, min_area=6):
    mask = color_mask(img, COLORS[color_name])
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] < min_area:
            continue
        cx, cy = centroids[lbl]
        d = float(np.hypot(cx - pred_col, cy - pred_row))
        if best is None or d < best[0]:
            best = (d, cx, cy)
    return best  # (dist, cx, cy) or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resets", type=int, default=20)
    args = ap.parse_args()

    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward, control_frequency=10.0, seed=7)
    order = list(blocks.FIXED_8_COMBINATION)

    all_err, matched, total = [], 0, 0
    per_color = collections.defaultdict(lambda: [0, 0])  # color -> [matched, total]
    eff_changed = []
    annot_saved = False
    for r in range(args.resets):
        env.reset()
        state = env.compute_state()
        rgb_vis = lt_render.render_topdown(env, size=SIZE, hide_effector=False)
        rgb_hidden = lt_render.render_topdown(env, size=SIZE, hide_effector=True)
        eff_changed.append(int((np.abs(rgb_vis.astype(int) - rgb_hidden.astype(int)).sum(2) > 30).sum()))

        if not annot_saved:
            annot = cv2.cvtColor(rgb_hidden.copy(), cv2.COLOR_RGB2BGR)
        for b in order:
            color = b.split("_")[0]
            x, y = np.asarray(state[f"block_{b}_translation"]).ravel()[:2]
            col, row = lt_render.world_to_pixel(x, y, SIZE)
            res = nearest_blob_dist(rgb_hidden, color, col, row)
            total += 1
            per_color[color][1] += 1
            if res is not None:
                d, cx, cy = res
                all_err.append(d)
                ok = d < MATCH_PX
                matched += int(ok)
                per_color[color][0] += int(ok)
                if not annot_saved:
                    cv2.drawMarker(annot, (int(col), int(row)), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
                    cv2.circle(annot, (int(cx), int(cy)), 3, (0, 255, 255), 1)
        if not annot_saved:
            cv2.imwrite("/tmp/lt_fidelity_annot.png", annot)
            cv2.imwrite("/tmp/lt_topdown_visible.png", cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
            cv2.imwrite("/tmp/lt_topdown_hidden.png", cv2.cvtColor(rgb_hidden, cv2.COLOR_RGB2BGR))
            annot_saved = True

    err = np.array(all_err)
    print(f"resets={args.resets}  blocks={total}")
    print(f"FIDELITY match-rate (<{MATCH_PX:.0f}px): {matched}/{total} = {matched/total:.3f}")
    print(f"pixel error: mean={err.mean():.2f}  median={np.median(err):.2f}  "
          f"p90={np.percentile(err,90):.2f}  max={err.max():.2f}  "
          f"(world: mean={err.mean()/PX_PER_UNIT*1000:.1f}mm)")
    for c in COLORS:
        m, t = per_color[c]
        print(f"  {c:7s} {m}/{t} = {m/t:.2f}")
    print(f"EFFECTOR-HIDE px changed: mean={np.mean(eff_changed):.0f} "
          f"min={min(eff_changed)} max={max(eff_changed)}")
    print("saved /tmp/lt_fidelity_annot.png, /tmp/lt_topdown_{visible,hidden}.png")


if __name__ == "__main__":
    main()
