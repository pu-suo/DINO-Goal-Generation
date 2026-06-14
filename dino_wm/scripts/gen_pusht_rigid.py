"""Generate the language-conditioned rigid-transform PushT dataset (Part 1).

For each eligible REAL pusht_noise trajectory: exclude wall-users (1.1), sample a
full-path-valid rigid transform (1.2/1.4) that places the goal block uniformly
(1.8), render the CLEAN (green-T-removed, 1.5) start & goal frames on the FIXED
scene (1.3), and emit the (start_frame, goal_frame, language) triple (1.7).

Train/test use DISJOINT trajectory pools, so no underlying real push appears in
both (no pairing leakage). Saves states + clean visuals + language + labels;
encoding to DINOv2 latents is a separate GPU step (cache_rigid_latents.py).

  DATASET_DIR=/workspace/data python scripts/gen_pusht_rigid.py \
    --data_path /workspace/data/pusht_noise --src_split train \
    --out /workspace/data/pusht_rigid --n_train 8000 --n_test 1500
"""
import os, sys, argparse, pickle, json
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.rigid_goal_common import (
    traj_uses_wall, sample_valid_transform, make_language,
    assert_relative_geometry_preserved, REGION_BOUNDS,
)
from datasets.rigid_goal_render import make_env, render_state
from metrics.regional_success import block_cell, REGION_NCELLS


def build_split(states, seq, traj_ids, n_target, rng, env, wall_margin, min_seq, save_visual):
    rec = {k: [] for k in ("start_state", "goal_state", "start_vis", "goal_vis",
                            "lang", "region_cell", "rel_rot", "goal_pusher_xy")}
    used = excl_wall = excl_short = excl_nofit = 0
    for i in traj_ids:
        if len(rec["start_state"]) >= n_target:
            break
        L = int(seq[i])
        if L < min_seq:
            excl_short += 1; continue
        if traj_uses_wall(states[i], L, wall_margin):
            excl_wall += 1; continue
        res = sample_valid_transform(states[i], L, rng, goal_bounds=REGION_BOUNDS)
        if res is None:
            excl_nofit += 1; continue
        theta, t, tf = res
        assert_relative_geometry_preserved(states[i], tf, L)   # rigid sanity (raises if broken)
        s0, sT = tf[0].copy(), tf[L - 1].copy()
        lang = make_language(s0, sT)
        rec["start_state"].append(s0); rec["goal_state"].append(sT)
        rec["lang"].append(lang["text"]); rec["region_cell"].append(lang["region_cell"])
        rec["rel_rot"].append(lang["rel_rot_deg"]); rec["goal_pusher_xy"].append(sT[0:2])
        if save_visual:
            rec["start_vis"].append(render_state(env, s0)[0])
            rec["goal_vis"].append(render_state(env, sT)[0])
        used += 1
    out = {
        "start_states": torch.tensor(np.array(rec["start_state"]), dtype=torch.float32),
        "goal_states": torch.tensor(np.array(rec["goal_state"]), dtype=torch.float32),
        "region_cells": torch.tensor(np.array(rec["region_cell"]), dtype=torch.int64),
        "rel_rots": torch.tensor(np.array(rec["rel_rot"]), dtype=torch.float32),
        "goal_pusher_xy": torch.tensor(np.array(rec["goal_pusher_xy"]), dtype=torch.float32),
        "languages": rec["lang"],
    }
    if save_visual:
        out["start_visual"] = torch.tensor(np.array(rec["start_vis"]), dtype=torch.uint8)
        out["goal_visual"] = torch.tensor(np.array(rec["goal_vis"]), dtype=torch.uint8)
    stats = {"n": used, "excl_wall": excl_wall, "excl_short": excl_short, "excl_nofit": excl_nofit}
    return out, stats


def save_split(out_dir, split, data):
    d = os.path.join(out_dir, split); os.makedirs(d, exist_ok=True)
    for k, v in data.items():
        if k == "languages":
            pickle.dump(v, open(os.path.join(d, "languages.pkl"), "wb"))
        else:
            torch.save(v, os.path.join(d, f"{k}.pth"))


def sample_grid_visual(data, out_png, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from metrics.regional_success import region_name
    n = min(n, data["start_states"].shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(5.2, 2.6 * n))
    for r in range(n):
        for c, key in enumerate(("start_visual", "goal_visual")):
            ax = axes[r, c]
            ax.imshow(data[key][r].numpy()); ax.axis("off")
            if c == 0:
                ax.set_title("start (clean)", fontsize=8)
            else:
                cell = tuple(data["region_cells"][r].tolist())
                ax.set_title(f"goal: {region_name(cell)}, {data['rel_rots'][r]:.0f}deg", fontsize=8)
        axes[r, 0].text(-0.05, 0.5, f"#{r}", transform=axes[r, 0].transAxes,
                        va="center", ha="right", fontsize=8)
    fig.suptitle("Rigid-transform language goals: start (clean) -> goal (clean)\n"
                 "+ generated language", fontsize=10)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"saved sample grid -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--src_split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--wall_margin", type=float, default=10.0)
    ap.add_argument("--min_seq", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_visual", action="store_true", help="skip rendering (states/lang only)")
    ap.add_argument("--grid_png", default=None)
    args = ap.parse_args()

    d = os.path.join(args.data_path, args.src_split)
    states = torch.load(os.path.join(d, "states.pth")).double().numpy()
    seq = pickle.load(open(os.path.join(d, "seq_lengths.pkl"), "rb"))
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(seq))
    # disjoint trajectory pools: first chunk -> test, rest -> train (no shared push)
    test_ids = order[:len(order) // 5]
    train_ids = order[len(order) // 5:]
    env = None if args.no_visual else make_env(with_target=False)

    print(f"generating from {len(seq)} {args.src_split} trajs "
          f"(pools: {len(train_ids)} train / {len(test_ids)} test, disjoint)")
    tr, tr_s = build_split(states, seq, train_ids, args.n_train, rng, env,
                           args.wall_margin, args.min_seq, not args.no_visual)
    te, te_s = build_split(states, seq, test_ids, args.n_test, rng, env,
                           args.wall_margin, args.min_seq, not args.no_visual)
    save_split(args.out, "train", tr); save_split(args.out, "test", te)
    manifest = {"n_train": tr_s["n"], "n_test": te_s["n"], "train_excl": tr_s,
                "test_excl": te_s, "seed": args.seed, "goal_bounds": list(REGION_BOUNDS),
                "ncells": REGION_NCELLS, "disjoint_traj_pools": True}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"train n={tr_s['n']} (excl {tr_s}) | test n={te_s['n']} (excl {te_s})")
    print(f"-> {args.out}")
    if not args.no_visual:
        grid = args.grid_png or os.path.join(args.out, "sample_grid.png")
        sample_grid_visual(tr, grid, n=6)


if __name__ == "__main__":
    main()
