"""Confirm the manipulator mask lands on the pusher (row-major patch ordering) on a
REAL frame. Builds manipulator_energy_mask() for each eval's goal frame and overlays the
DROPPED patches (red tint) on obs_g, plus a marker at the pusher's sim->image position.
If ordering/geometry are right, the red patches sit squarely on the blue pusher circle.

Run on the box (where a plan_targets.pkl lives) or locally after scp:
  python analysis/verify_pusher_mask.py plan_outputs/<dir>/plan_targets.pkl \
      --evals 0,1,2 --dilation 0 --out docs/mask_check
"""
import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.pusht.multicolor_common import manipulator_energy_mask  # noqa: E402

GRID = 14


def to_img(arr):
    a = np.squeeze(np.asarray(arr))
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255 if a.max() <= 1.0 else a).clip(0, 255).astype(np.uint8)
    return a


def outline_drops(ax, keep_mask, h, w):
    """Draw red OUTLINES around DROPPED patches (image left untouched underneath, so the
    blue pusher stays fully visible). Mask is row-major (ri*GRID + ci)."""
    import matplotlib.patches as mpatches
    ph, pw = h / GRID, w / GRID
    drop = (keep_mask == 0.0).reshape(GRID, GRID)
    for ri in range(GRID):
        for ci in range(GRID):
            if drop[ri, ci]:
                ax.add_patch(mpatches.Rectangle(
                    (ci * pw - 0.5, ri * ph - 0.5), pw, ph,
                    fill=False, edgecolor="red", linewidth=1.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--evals", default="0,1,2", help="'all' or e.g. 0,1,2")
    ap.add_argument("--dilation", type=int, default=0)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="docs/mask_check")
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, "rb"))
    sg = np.asarray(d["state_g"], float)        # real pusher at [:, 0:2]
    vis = d["obs_g"]["visual"]
    n = len(sg)
    idx = list(range(n)) if args.evals == "all" else [int(x) for x in args.evals.split(",")]
    os.makedirs(args.out, exist_ok=True)

    # 2 columns per eval: [clean obs_g | obs_g with red patch-outlines + cyan pusher +]
    fig, axes = plt.subplots(len(idx), 2, figsize=(6.4, 3.2 * len(idx)))
    axes = np.atleast_2d(axes)
    for row, i in enumerate(idx):
        img = to_img(vis[i])
        h, w = img.shape[:2]
        real_p = sg[i, 0:2]
        keep = manipulator_energy_mask([real_p], dilation=args.dilation)
        px, py = real_p[0] * w / 512.0, real_p[1] * h / 512.0
        for col, ax in enumerate(axes[row]):
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 1:
                outline_drops(ax, keep, h, w)                 # red boxes, image intact
                ax.plot(px, py, "c+", ms=14, mew=2.5)         # independent pusher marker
        axes[row][0].set_ylabel(f"seed {args.seed*i+1}\ndrop={int((keep==0).sum())} patches",
                                fontsize=9, rotation=0, ha="right", va="center", labelpad=28)
        if row == 0:
            axes[row][0].set_title("obs_g (clean)", fontsize=9)
            axes[row][1].set_title("dropped patches (red) + pusher (cyan +)", fontsize=9)
    plt.tight_layout()
    p = os.path.join(args.out, f"mask_check_dil{args.dilation}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"wrote {p}  ({len(idx)} evals). Red patches should cover the blue pusher (cyan +).")


if __name__ == "__main__":
    main()
