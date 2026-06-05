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


def overlay(img, keep_mask, sim=512):
    """Tint DROPPED patches red on a HxW image; mask is row-major (ri*GRID + ci)."""
    img = img.copy()
    h, w = img.shape[:2]
    ph, pw = h / GRID, w / GRID
    drop = (keep_mask == 0.0).reshape(GRID, GRID)
    for ri in range(GRID):
        for ci in range(GRID):
            if drop[ri, ci]:
                r0, r1 = int(ri * ph), int((ri + 1) * ph)
                c0, c1 = int(ci * pw), int((ci + 1) * pw)
                patch = img[r0:r1, c0:c1].astype(np.float32)
                patch[..., 0] = 0.6 * patch[..., 0] + 0.4 * 255  # push red
                patch[..., 1] *= 0.6
                patch[..., 2] *= 0.6
                img[r0:r1, c0:c1] = patch.clip(0, 255).astype(np.uint8)
    return img


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

    fig, axes = plt.subplots(len(idx), 1, figsize=(3.2, 3.2 * len(idx)))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idx):
        img = to_img(vis[i])
        h = img.shape[0]
        real_p = sg[i, 0:2]
        keep = manipulator_energy_mask([real_p], dilation=args.dilation)
        ov = overlay(img, keep)
        ax.imshow(ov)
        # marker at the pusher's sim->image position (should be under the red patches)
        ax.plot(real_p[0] * h / 512.0, real_p[1] * h / 512.0, "c+", ms=12, mew=2)
        ax.set_title(f"seed {args.seed*i+1}  drop={int((keep==0).sum())} patches",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    p = os.path.join(args.out, f"mask_check_dil{args.dilation}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"wrote {p}  ({len(idx)} evals). Red patches should cover the blue pusher (cyan +).")


if __name__ == "__main__":
    main()
