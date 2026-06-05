"""Render [start | real-goal | contact-goal] triptychs to eyeball the procedural
pusher placement against the true recorded pusher, for the same tasks.

The rendered goal frames live in each run's `plan_targets.pkl` (state_g keeps the
REAL pusher; obs_g is whatever pusher that run rendered). So we stitch the stored
frames from BOTH the `real` run and the `contact` run, paired by eval index:
  start       = real_run.obs_0[i]      (identical across conditions)
  real-goal   = real_run.obs_g[i]      (true recorded pusher)
  contact-goal= contact_run.obs_g[i]   (procedural pusher)
and annotate the real-vs-contact pusher gap + block rotation.

Copy the two pkls from the GPU box to your Mac first, e.g.:
  scp box:/workspace/dino_goal/dino_wm/plan_outputs/<REAL_dir>/plan_targets.pkl    real.pkl
  scp box:/workspace/dino_goal/dino_wm/plan_outputs/<CONTACT_dir>/plan_targets.pkl contact.pkl
Then:
  python analysis/render_pusher_compare.py --real_pkl real.pkl --contact_pkl contact.pkl \
      --evals 1,2,3,6 --out docs/pusher_compare \
      --success_real "T,T,T,T,T,T,T,T,T,T" --success_contact "T,F,F,F,T,T,F,T,T,T"
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.pusht.multicolor_common import contact_pusher_pose, angle_diff  # noqa: E402


def to_img(arr):
    a = np.squeeze(np.asarray(arr))
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))          # CHW -> HWC
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.dtype != np.uint8:
        a = (a * 255 if a.max() <= 1.0 else a).clip(0, 255).astype(np.uint8)
    return a


def parse_mask(s, n):
    if s is None:
        return None
    vals = [t.lower() in ("true", "t", "1")
            for t in re.findall(r"[A-Za-z]+|[01]", s)
            if t.lower() in ("true", "t", "1", "false", "f", "0")]
    return np.array(vals, dtype=bool) if len(vals) == n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_pkl", required=True)
    ap.add_argument("--contact_pkl", required=True)
    ap.add_argument("--evals", default="all", help="'all' or e.g. 1,2,3,6")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="docs/pusher_compare")
    ap.add_argument("--success_real", default=None)
    ap.add_argument("--success_contact", default=None)
    ap.add_argument("--per_task", action="store_true",
                    help="also write one PNG per eval (default: just the combined sheet)")
    args = ap.parse_args()

    R = pickle.load(open(args.real_pkl, "rb"))
    C = pickle.load(open(args.contact_pkl, "rb"))
    s0 = np.asarray(R["state_0"], float)
    sg = np.asarray(R["state_g"], float)      # real state (real pusher at [0:2])
    n = len(s0)
    mr = parse_mask(args.success_real, n)
    mc = parse_mask(args.success_contact, n)
    idx = list(range(n)) if args.evals == "all" else [int(x) for x in args.evals.split(",")]
    os.makedirs(args.out, exist_ok=True)

    fig, axes = plt.subplots(len(idx), 3, figsize=(9, 3 * len(idx)))
    axes = np.atleast_2d(axes)
    col_titles = ["start", "real-goal (true pusher)", "contact-goal (procedural)"]
    for row, i in enumerate(idx):
        block0, blockg, thg, th0 = s0[i, 2:4], sg[i, 2:4], sg[i, 4], s0[i, 4]
        real_p = sg[i, 0:2]
        cp = contact_pusher_pose(block0, np.r_[blockg, thg])
        gap = float(np.linalg.norm(real_p - cp)) if cp is not None else 0.0
        rot = float(np.degrees(angle_diff(thg, th0)))
        trans = float(np.linalg.norm(blockg - block0))
        imgs = [to_img(R["obs_0"]["visual"][i]),
                to_img(R["obs_g"]["visual"][i]),
                to_img(C["obs_g"]["visual"][i])]
        for col, (ax, im) in enumerate(zip(axes[row], imgs)):
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            # color the contact-goal panel border by success so pass/fail scans easily
            if col == 2 and mc is not None:
                edge = "green" if mc[i] else "red"
                for sp in ax.spines.values():
                    sp.set_edgecolor(edge); sp.set_linewidth(4)
        flags = ""
        if mr is not None and mc is not None:
            flags = f"  contact {'PASS' if mc[i] else 'FAIL'}"
        axes[row][0].set_ylabel(
            f"seed {args.seed*i+1}\ntrans {trans:.0f}px\nrot {rot:.0f}°\n"
            f"pusher gap {gap:.0f}px{flags}",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=40)
    plt.tight_layout()
    sheet = os.path.join(args.out, "pusher_compare.png")
    fig.savefig(sheet, dpi=130, bbox_inches="tight")
    print(f"wrote {sheet}  ({len(idx)} tasks)")

    # optionally, one PNG per eval for close inspection
    if args.per_task:
        for row, i in enumerate(idx):
            f2, ax2 = plt.subplots(1, 3, figsize=(9, 3.4))
            for col in range(3):
                ax2[col].imshow(to_img([R, R, C][col]["obs_" + ("0" if col == 0 else "g")]["visual"][i]))
                ax2[col].set_xticks([]); ax2[col].set_yticks([]); ax2[col].set_title(col_titles[col], fontsize=10)
            p = os.path.join(args.out, f"eval{i}_seed{args.seed*i+1}.png")
            f2.savefig(p, dpi=130, bbox_inches="tight"); plt.close(f2)
        print(f"wrote {len(idx)} per-task PNGs to {args.out}/")


if __name__ == "__main__":
    main()
