"""
Visualize multi-color PushT *tasks*: for each sampled layout, render the START
frame (block + pusher + N colored target T's) next to the GOAL frame (block moved
onto the NAMED target). The start frame is ambiguous by design -- the instruction
text is the only cue -- so the goal frame is what reveals which target was named.

    cd dino_wm
    SDL_VIDEODRIVER=dummy PYTHONPATH=. python scripts/viz_tasks.py \
        --n 5 --seed 0 --combo_split all --out /tmp/mc_tasks.png

Pure local (Mac dev) -- no model/dataset needed, just the env.
"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import numpy as np
import cv2

from env.pusht.pusht_multicolor_env import PushTMultiColorEnv
from env.pusht import multicolor_sampler as mcs


def _render(env, full_state):
    env.seed(0)
    env.reset_to_state = np.asarray(full_state, dtype=np.float64).copy()
    obs, _ = env.reset()
    return np.ascontiguousarray(obs["visual"])  # (S,S,3) uint8 RGB


def _band(text, w, h=26, color=(20, 20, 20), bg=(255, 255, 255), scale=0.5):
    b = np.full((h, w, 3), bg, np.uint8)
    cv2.putText(b, text, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of tasks to show")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_targets", type=int, default=4)
    ap.add_argument("--render_size", type=int, default=256)
    ap.add_argument("--combo_split", default="all", choices=["all", "train", "heldout"])
    ap.add_argument("--max_goal_dist", type=float, default=None)
    ap.add_argument("--max_goal_angle", type=float, default=None)
    ap.add_argument("--out", default="viz_outputs/tasks.png",
                    help="output PNG (relative paths land under dino_wm/ when run from there)")
    args = ap.parse_args()

    allowed = active = None
    if args.combo_split in ("train", "heldout"):
        train, test = mcs.make_combo_split(args.n_targets)
        allowed = train
        active = test if args.combo_split == "heldout" else None

    env = PushTMultiColorEnv(with_velocity=True, n_targets=args.n_targets,
                             render_size=args.render_size)
    S = args.render_size
    rows = [_band("START  (4 targets visible; text is the only cue)", S, 24, scale=0.42)]
    rows[0] = np.concatenate(
        [rows[0], _band("GOAL  (block on the NAMED target)", S, 24, scale=0.42)], axis=1)

    for i in range(args.n):
        lay = mcs.sample_layout(args.seed * 1000 + i, n_targets=args.n_targets,
                                allowed_combos=allowed, active_combos=active,
                                max_goal_dist=args.max_goal_dist,
                                max_goal_angle=args.max_goal_angle)
        env.set_layout(lay)
        init = lay["init_state"]
        goal = init.copy(); goal[2:5] = lay["goal_pose"]   # block -> named target
        start_img = _render(env, init)
        goal_img = _render(env, goal)

        pair = np.concatenate([start_img, goal_img], axis=1)
        colors = ", ".join(t["color"] for t in lay["targets"])
        label = _band(f'"{lay["instruction"]}"   |   named={lay["active_color"]}   '
                      f'(targets: {colors})', pair.shape[1], 26, scale=0.45)
        rows.append(label)
        rows.append(pair)
        rows.append(np.full((3, pair.shape[1], 3), 200, np.uint8))  # separator

    montage = np.concatenate(rows, axis=0)
    _os.makedirs(_os.path.dirname(_os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.out}  {montage.shape}  ({args.n} tasks, split={args.combo_split})")


if __name__ == "__main__":
    main()
