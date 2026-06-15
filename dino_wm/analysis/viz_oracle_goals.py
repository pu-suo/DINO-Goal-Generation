"""Visualize the actual oracle data: original PushT (green goal-T) vs the CLEAN
start (z_start) vs the CLEAN goal the oracle plans toward, one row per rotation
bucket. Shows the green-T removal (with_target False -> painted White-on-White) and
the start->goal block motion + rotation that the command describes.

  DATASET_DIR=/workspace/data python analysis/viz_oracle_goals.py \
    --data /workspace/data/pusht_subseg/test --out analysis/out/oracle_goals_viz.png
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys, pickle, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.rigid_goal_render import make_env, render_state
from datasets.rotation_command import bucket_name, all_buckets, signed_drot_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATASET_DIR", ".") + "/pusht_subseg/test")
    ap.add_argument("--out", default="analysis/out/oracle_goals_viz.png")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    d = Path(args.data)
    start = torch.load(d / "start_states.pth").numpy()
    goal = torch.load(d / "goal_states.pth").numpy()
    buckets = torch.load(d / "rot_buckets.pth").numpy()
    text = pickle.load(open(d / "command_text.pkl", "rb"))
    rng = np.random.RandomState(args.seed)

    # one example per bucket
    rows = []
    for b in all_buckets():
        idx = np.where((buckets[:, 0] == b[0]) & (buckets[:, 1] == b[1]))[0]
        if len(idx):
            rows.append(int(rng.choice(idx)))
    env_t = make_env(with_target=True)     # original: green goal-T visible
    env_c = make_env(with_target=False)    # clean: green-T painted white

    R = len(rows)
    fig, ax = plt.subplots(R, 3, figsize=(9.5, 3.2 * R))
    if R == 1:
        ax = ax[None, :]
    col_titles = ["original PushT\n(green goal-T visible, fixed center)",
                  "CLEAN start = z_start\n(green-T hidden, white-on-white)",
                  "CLEAN goal\n(oracle plans toward this)"]
    for r, k in enumerate(rows):
        s5, g5 = start[k].astype(np.float64), goal[k].astype(np.float64)
        img_orig = render_state(env_t, s5)[0]        # original start w/ green-T
        img_cs = render_state(env_c, s5)[0]          # clean start
        img_cg = render_state(env_c, g5)[0]          # clean goal
        drot = signed_drot_deg(s5[4], g5[4])
        disp = float(np.linalg.norm(g5[2:4] - s5[2:4]))
        for c, img in enumerate((img_orig, img_cs, img_cg)):
            ax[r, c].imshow(img); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(col_titles[c], fontsize=9)
        ax[r, 0].set_ylabel(f"{bucket_name(tuple(buckets[k]))}\nΔrot={drot:+.1f}°  Δpos={disp:.0f}px",
                            fontsize=9)
        # command under the clean-start panel (what g would receive)
        ax[r, 1].text(0.5, -0.08, f'"...{text[k]}"', transform=ax[r, 1].transAxes,
                      ha="center", va="top", fontsize=8, color="darkblue")
    fig.suptitle("Oracle data: green-T removal + start->goal (raw clean sub-segments, no rigid transform)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[viz] {R} rows -> {args.out}")
    # also dump the raw poses for reference
    for r, k in enumerate(rows):
        s5, g5 = start[k], goal[k]
        print(f"  {bucket_name(tuple(buckets[k])):14s} start block=({s5[2]:.0f},{s5[3]:.0f},{np.degrees(s5[4]):.0f}°) "
              f"pusher=({s5[0]:.0f},{s5[1]:.0f}) -> goal block=({g5[2]:.0f},{g5[3]:.0f},{np.degrees(g5[4]):.0f}°)")


if __name__ == "__main__":
    main()
