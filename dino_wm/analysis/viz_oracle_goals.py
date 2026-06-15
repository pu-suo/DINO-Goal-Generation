"""Visualize the actual oracle data CLEARLY: clean START vs clean GOAL vs an
OVERLAY (start=blue, goal=red) so the commanded block move+rotation is visible even
when small. The stock green goal-T is REMOVED and is irrelevant to our task -- our
goal is the BLOCK's target pose (gray T in the GOAL column / red in the overlay),
NOT any green marker.

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


def nonwhite(img):
    return img.astype(int).sum(2) < 740          # block/pusher pixels (not white bg)


def overlay(start, goal):
    """start block/pusher tinted BLUE, goal tinted RED, on white -> shows the move."""
    ov = np.full_like(start, 255)
    ms, mg = nonwhite(start), nonwhite(goal)
    ov[ms] = (ov[ms] * 0.30 + np.array([50, 90, 230]) * 0.70).astype(np.uint8)
    ov[mg] = (ov[mg] * 0.40 + np.array([230, 60, 50]) * 0.60).astype(np.uint8)
    return ov


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

    # per bucket: pick the example with the LARGEST motion (clearest to see; still in-bucket)
    rows = []
    for b in all_buckets():
        idx = np.where((buckets[:, 0] == b[0]) & (buckets[:, 1] == b[1]))[0]
        if not len(idx):
            continue
        disp = np.linalg.norm(goal[idx, 2:4] - start[idx, 2:4], axis=1)
        drot = np.abs([signed_drot_deg(start[k, 4], goal[k, 4]) for k in idx])
        score = disp + 4 * drot                  # weight rotation so it's visible too
        rows.append(int(idx[np.argmax(score)]))
    env_c = make_env(with_target=False)          # clean: green-T removed

    R = len(rows)
    fig, ax = plt.subplots(R, 3, figsize=(9.5, 3.3 * R))
    if R == 1:
        ax = ax[None, :]
    titles = ["CLEAN START\n(gray T = block now; this is z_start)",
              "CLEAN GOAL\n(gray T = block TARGET pose)",
              "OVERLAY  (start=blue, goal=red)\nthe commanded move + rotation"]
    for r, k in enumerate(rows):
        s5, g5 = start[k].astype(np.float64), goal[k].astype(np.float64)
        cs = render_state(env_c, s5)[0]
        cg = render_state(env_c, g5)[0]
        ov = overlay(cs, cg)
        drot = signed_drot_deg(s5[4], g5[4])
        disp = float(np.linalg.norm(g5[2:4] - s5[2:4]))
        for c, img in enumerate((cs, cg, ov)):
            ax[r, c].imshow(img); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(titles[c], fontsize=9)
        ax[r, 0].set_ylabel(f"{bucket_name(tuple(buckets[k]))}\nΔrot={drot:+.1f}°  Δpos={disp:.0f}px",
                            fontsize=9)
        ax[r, 2].text(0.5, -0.07, f'command: "...{text[k]}"', transform=ax[r, 2].transAxes,
                      ha="center", va="top", fontsize=8, color="darkblue")
    fig.suptitle("Oracle data (raw clean sub-segments). Goal = the BLOCK's target pose; "
                 "the stock green goal-T is REMOVED (irrelevant).", fontsize=10, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[viz] {R} rows -> {args.out}")
    for r, k in enumerate(rows):
        s5, g5 = start[k], goal[k]
        print(f"  {bucket_name(tuple(buckets[k])):14s} block ({s5[2]:.0f},{s5[3]:.0f},{np.degrees(s5[4]):.0f}°)"
              f" -> ({g5[2]:.0f},{g5[3]:.0f},{np.degrees(g5[4]):.0f}°)  "
              f"Dpos={np.linalg.norm(g5[2:4]-s5[2:4]):.0f}px Drot={signed_drot_deg(s5[4],g5[4]):+.1f}°")


if __name__ == "__main__":
    main()
