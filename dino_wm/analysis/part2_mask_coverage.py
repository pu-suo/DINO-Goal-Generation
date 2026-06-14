"""Part 2: confirm the deployable pusher energy-mask transfers to the NEW rigid
goal frames. The mask drops the goal-frame pusher's patches (manipulator_energy_mask
over goal_pusher_xy); with objective.alpha=0 the proprio term is off, so the
goal-time pusher is fully mooted. We verify the dropped patches OVERLAP the pusher
in each new goal frame and visualize the coverage.

  python analysis/part2_mask_coverage.py --data <pusht_rigid or smoke> --split test
"""
import os, sys, argparse
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.rigid_goal_render import make_env, render_state
from env.pusht.multicolor_common import manipulator_energy_mask, pusher_patch_mask

GRID, IMG = 14, 224
SIM = 512.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="_devdata/pusht_rigid_smoke")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dilation", type=int, default=1)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="analysis_outputs/part2_mask_coverage.png")
    args = ap.parse_args()

    d = os.path.join(args.data, args.split)
    goal_states = torch.load(os.path.join(d, "goal_states.pth")).numpy()
    goal_pusher = torch.load(os.path.join(d, "goal_pusher_xy.pth")).numpy()
    env = make_env(with_target=True)  # 3.1 scene (green-T present, inert)

    n = min(args.n, len(goal_states))
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    n_covered = 0
    for i in range(n):
        gs = goal_states[i]
        pxy = goal_pusher[i]
        img, _ = render_state(env, gs)
        keep = manipulator_energy_mask([pxy], dilation=args.dilation)   # (196,) 1=keep/0=drop
        drop = (keep == 0).reshape(GRID, GRID)                          # [row, col] = [y, x]
        # which patch holds the pusher? (col<->x, row<->y on the 196-grid)
        pcol = int(np.clip(pxy[0] * 196 / SIM // 14, 0, GRID - 1))
        prow = int(np.clip(pxy[1] * 196 / SIM // 14, 0, GRID - 1))
        covered = bool(drop[prow, pcol]); n_covered += covered
        ax = axes[i]
        ax.imshow(img)
        ps = IMG / GRID
        for r in range(GRID):
            for c in range(GRID):
                if drop[r, c]:
                    ax.add_patch(Rectangle((c * ps, r * ps), ps, ps, facecolor="red",
                                           alpha=0.32, edgecolor="none"))
        ax.plot(pxy[0] * IMG / SIM, pxy[1] * IMG / SIM, "c+", ms=12, mew=2)
        ax.set_title(f"#{i} pusher patch dropped: {covered}\n{int(drop.sum())}/196 dropped",
                     fontsize=8)
        ax.axis("off")
    fig.suptitle("Part 2: deployable pusher mask (red=dropped patches) over rigid goal frames\n"
                 "cyan + = goal-frame pusher (alpha=0 -> proprio off; masked visual -> pusher mooted)",
                 fontsize=9)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"pusher patch covered by mask: {n_covered}/{n}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
