"""
Phase 0.1 local sanity preview for the multi-color PushT env.

Renders a handful of layouts (start frame + "block at named target" goal frame),
saves a labeled montage, and prints the decorrelation statistics. Meant to be
eyeballed on the Mac before any GPU work.

    cd dino_wm
    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python scripts/verify_multicolor_env.py
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.pusht.pusht_multicolor_env import PushTMultiColorEnv
from env.pusht import multicolor_sampler as mcs


def render_layout(env, layout):
    """Return (start_frame, goal_frame) for a layout."""
    env.set_layout(layout)
    init = layout["init_state"]
    env.seed(0); env.reset_to_state = init.copy()
    start, _ = env.reset()

    goal_state = init.copy()
    goal_state[2:5] = layout["goal_pose"]
    env.seed(0); env.reset_to_state = goal_state.copy()
    goal, _ = env.reset()
    return start["visual"], goal["visual"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="layouts to preview")
    ap.add_argument("--n_targets", type=int, default=4)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--outline_thickness", type=int, default=7)
    ap.add_argument("--out", default="analysis_outputs/multicolor_env_preview.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    env = PushTMultiColorEnv(
        with_velocity=True, n_targets=args.n_targets,
        render_size=args.render_size, outline_thickness=args.outline_thickness,
    )

    fig, axes = plt.subplots(args.n, 2, figsize=(6, 3 * args.n))
    if args.n == 1:
        axes = axes[None, :]
    for i in range(args.n):
        layout = mcs.sample_layout(1000 + i, n_targets=args.n_targets)
        start, goal = render_layout(env, layout)
        axes[i, 0].imshow(start); axes[i, 0].axis("off")
        axes[i, 0].set_title(f'start | "{layout["instruction"]}"', fontsize=8)
        axes[i, 1].imshow(goal); axes[i, 1].axis("off")
        axes[i, 1].set_title(
            f'goal: block@{layout["active_color"]} '
            f'({layout["goal_pose"][0]:.0f},{layout["goal_pose"][1]:.0f})', fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved preview montage -> {args.out}")

    # decorrelation report
    rate, chance, n = mcs.nearest_target_predicts_named(
        n_samples=5000, n_targets=args.n_targets, seed=2)
    print(f"\nDecorrelation: P(named == nearest) = {rate:.4f}  (chance 1/{args.n_targets} = {chance:.4f}, n={n})")
    status = "OK" if abs(rate - chance) < 0.03 else "WARN: correlated!"
    print(f"  -> {status}")

    # split scaffold report
    train, test = mcs.make_combo_split(n_targets=args.n_targets, n_bins=3, heldout_frac=0.2, seed=0)
    print(f"\nColor-location split: {len(train)} train combos, {len(test)} held-out combos")
    print(f"  held-out examples: {sorted(list(test))[:6]}")

    # a few instructions
    print("\nSample instructions:")
    for i in range(5):
        lay = mcs.sample_layout(2000 + i, n_targets=args.n_targets)
        print(f'  [{lay["active_color"]:7s}] "{lay["instruction"]}"  (template {lay["template_id"]})')


if __name__ == "__main__":
    main()
