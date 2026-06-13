"""
Phase-1 (clean-scene pivot, Option A): generate the single-T COORDINATE-spec PushT
dataset on the CLEAN-AS-STOCK scene.

Scene = the stock base PushTEnv render: a CONSTANT green goal-T at the fixed center
pose [256,256,pi/4] + the gray block + the blue pusher, on white. NO colored decals.
The constant green T is INERT for `g`: it is identical in the start frame, the goal
frame, and every planner rollout frame, so it cancels in the residual (z_goal-z_start)
and contributes ~0 to the deployable masked latent-L2 energy. This is why Option A
reuses the proven stock single-T dynamics (block tf-1step ~8.0, plans @0.90) and the
single-T pose-decode probe (~5.4px/4.4deg) untouched -- see docs/CLEAN_SCENE_PIVOT.md.

The goal is specified by a freely-sampled coordinate (x,y,theta) that is DECORRELATED
from the start block pose (sampled independently over the workspace), so `g` cannot
shortcut by predicting a typical delta -- it must read the spec. The goal frame
teleports the block to (x,y,theta) with the PUSHER LEFT AT ITS START position, so the
only thing that changes between start and goal is the block (pusher + green-T unchanged).
This matches the masked energy, which ignores the (goal-time-unknown) pusher.

Output (per split) under <out>/<split>/:
    start_obses/episode_XXXXXX.png    # render of the start state (block@A, pusher@start)
    goal_obses/episode_XXXXXX.png     # teleport render (block@B, pusher@start)
    init_states.pth   (N, sd)         # start state [ax,ay,bx,by,theta(,vx,vy)] sim-512
    specs.pth         (N, 3)          # goal pose (x,y,theta) sim-512  == the coordinate spec (block@B)
    start_poses.pth   (N, 3)          # start block pose (bx,by,theta) sim-512  (for decorrelation checks)
    ab_dist.pth       (N,)            # ||B_xy - A_xy|| sim-px (for reachability subsetting at Stage-2)
    labels.pkl        list[dict]      # {goal_pose, init_state, start_pose, seed}
plus <out>/coord_manifest.json (params + decorrelation stats).

Smoke (Mac dev env):
    cd dino_wm
    SDL_VIDEODRIVER=dummy /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python \
        scripts/gen_pusht_coord.py --out data/pusht_coord_smoke --n_train 8 --n_val 2 --n_test 4 --workers 1
Real run (vast.ai 4090, CPU-parallel):
    cd dino_wm && DATASET_DIR=/workspace/data /workspace/envs/dino_wm/bin/python \
        scripts/gen_pusht_coord.py --out $DATASET_DIR/pusht_coord --n_train 6000 --n_val 400 --n_test 1000 --workers 24
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame in every worker

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import imageio

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from env.pusht.pusht_env import PushTEnv

# Workspace bounds (sim-512). Block centers are kept in [BLOCK_LO, BLOCK_HI] so the
# whole T stays on-frame; matches the multicolor decal range (120..392) loosely and
# the stock env block-init range (100..400).
BLOCK_LO, BLOCK_HI = 110.0, 402.0
AGENT_LO, AGENT_HI = 60.0, 452.0
# Reject a goal pose whose center is within this of the start pusher (avoid the
# teleported block visually overlapping the pusher in the goal frame).
MIN_GOAL_PUSHER_SEP = 55.0


def sample_pose(rng):
    return np.array([rng.uniform(BLOCK_LO, BLOCK_HI),
                     rng.uniform(BLOCK_LO, BLOCK_HI),
                     rng.uniform(0.0, 2 * np.pi)], dtype=np.float64)


def sample_episode_layout(seed):
    """Independent start state + decorrelated goal pose. The goal block pose B is
    drawn INDEPENDENTLY of the start block pose A (full decorrelation), rejecting only
    goals that would place the block on top of the start pusher."""
    rng = np.random.RandomState(seed)
    agent_xy = np.array([rng.uniform(AGENT_LO, AGENT_HI),
                         rng.uniform(AGENT_LO, AGENT_HI)], dtype=np.float64)
    start_pose = sample_pose(rng)               # block A
    for _ in range(64):
        goal_pose = sample_pose(rng)            # block B, independent of A
        if np.linalg.norm(goal_pose[:2] - agent_xy) >= MIN_GOAL_PUSHER_SEP:
            break
    init_state = np.array([agent_xy[0], agent_xy[1],
                           start_pose[0], start_pose[1], start_pose[2],
                           0.0, 0.0], dtype=np.float32)   # with_velocity -> 7-dim
    return {"seed": int(seed), "init_state": init_state,
            "start_pose": start_pose.astype(np.float32),
            "goal_pose": goal_pose.astype(np.float32)}


def render_state(env, full_state7):
    env.reset_to_state = np.asarray(full_state7, dtype=np.float32).copy()
    obs, _ = env.reset()
    return obs["visual"]  # (H,W,3) uint8


def generate_one(task):
    idx, layout, render_size, split_dir = task
    env = PushTEnv(render_size=render_size, with_velocity=True, with_target=True)
    env.seed(int(layout["seed"]))

    init_state = np.asarray(layout["init_state"], dtype=np.float32)
    start_frame = render_state(env, init_state)

    # teleport goal frame: block -> goal_pose (x,y,theta), pusher left at start, vel 0
    goal_state = init_state.copy()
    goal_state[2:5] = np.asarray(layout["goal_pose"], dtype=np.float32)
    goal_frame = render_state(env, goal_state)

    imageio.imwrite(os.path.join(split_dir, "start_obses", f"episode_{idx:06d}.png"), start_frame)
    imageio.imwrite(os.path.join(split_dir, "goal_obses", f"episode_{idx:06d}.png"), goal_frame)

    label = {
        "goal_pose": np.asarray(layout["goal_pose"], dtype=np.float32),
        "init_state": init_state,
        "start_pose": np.asarray(layout["start_pose"], dtype=np.float32),
        "seed": int(layout["seed"]),
    }
    return idx, label


def build_split(name, layouts, out_root, render_size, workers):
    split_dir = os.path.join(out_root, name)
    os.makedirs(os.path.join(split_dir, "start_obses"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "goal_obses"), exist_ok=True)
    n = len(layouts)
    tasks = [(i, layouts[i], render_size, split_dir) for i in range(n)]
    results = [None] * n
    if workers <= 1:
        for t in tasks:
            r = generate_one(t)
            results[r[0]] = r
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(generate_one, t) for t in tasks]
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                results[r[0]] = r
                done += 1
                if done % max(1, n // 10) == 0 or done == n:
                    print(f"  [{name}] {done}/{n} episodes")

    labels = [r[1] for r in results]
    init_states = np.stack([lab["init_state"] for lab in labels]).astype(np.float32)
    specs = np.stack([lab["goal_pose"] for lab in labels]).astype(np.float32)
    start_poses = np.stack([lab["start_pose"] for lab in labels]).astype(np.float32)
    ab_dist = np.linalg.norm(specs[:, :2] - start_poses[:, :2], axis=1).astype(np.float32)

    import torch
    torch.save(torch.from_numpy(init_states), os.path.join(split_dir, "init_states.pth"))
    torch.save(torch.from_numpy(specs), os.path.join(split_dir, "specs.pth"))
    torch.save(torch.from_numpy(start_poses), os.path.join(split_dir, "start_poses.pth"))
    torch.save(torch.from_numpy(ab_dist), os.path.join(split_dir, "ab_dist.pth"))
    with open(os.path.join(split_dir, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)
    print(f"  [{name}] saved {n} episodes -> {split_dir}")
    return start_poses, specs, ab_dist


def decorrelation_stats(start_poses, specs):
    """Pearson corr between start block pose and goal (spec) pose. Independent
    sampling => ~0. Also the angle correlation and the A->B distance spread."""
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float((a * b).sum() / d)
    out = {
        "corr_x": pear(start_poses[:, 0], specs[:, 0]),
        "corr_y": pear(start_poses[:, 1], specs[:, 1]),
        "corr_theta": pear(np.cos(start_poses[:, 2]), np.cos(specs[:, 2])),
        "ab_dist_mean": float(np.linalg.norm(specs[:, :2] - start_poses[:, :2], axis=1).mean()),
        "ab_dist_p10": float(np.percentile(np.linalg.norm(specs[:, :2] - start_poses[:, :2], axis=1), 10)),
        "ab_dist_p90": float(np.percentile(np.linalg.norm(specs[:, :2] - start_poses[:, :2], axis=1), 90)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_coord"))
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Disjoint per-episode seed bands so splits never share an episode.
    train_layouts = [sample_episode_layout(args.seed + 1_000_000 + i) for i in range(args.n_train)]
    val_layouts = [sample_episode_layout(args.seed + 2_000_000 + i) for i in range(args.n_val)]
    test_layouts = [sample_episode_layout(args.seed + 3_000_000 + i) for i in range(args.n_test)]

    print(f"Generating coord splits (render={args.render_size}, workers={args.workers})...")
    sp_tr, sc_tr, _ = build_split("train", train_layouts, args.out, args.render_size, args.workers)
    build_split("val", val_layouts, args.out, args.render_size, args.workers)
    build_split("test", test_layouts, args.out, args.render_size, args.workers)

    dstats = decorrelation_stats(sp_tr, sc_tr)
    manifest = {
        "scene": "stock_single_T_clean (base PushTEnv, with_target=True, constant green goal-T, no decals)",
        "spec": "coordinate (x,y,theta) sim-512, block-goal; decorrelated (independent) from start block pose",
        "counts": {"train": args.n_train, "val": args.n_val, "test": args.n_test},
        "render_size": args.render_size, "seed": args.seed,
        "workspace": {"block_lo": BLOCK_LO, "block_hi": BLOCK_HI,
                      "agent_lo": AGENT_LO, "agent_hi": AGENT_HI,
                      "min_goal_pusher_sep": MIN_GOAL_PUSHER_SEP},
        "train_decorrelation": dstats,
    }
    with open(os.path.join(args.out, "coord_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Dataset at {args.out}")
    print(f"  train decorrelation (want ~0): corr_x={dstats['corr_x']:.3f} "
          f"corr_y={dstats['corr_y']:.3f} corr_theta={dstats['corr_theta']:.3f}")
    print(f"  A->B dist sim-px: mean={dstats['ab_dist_mean']:.0f} "
          f"p10={dstats['ab_dist_p10']:.0f} p90={dstats['ab_dist_p90']:.0f}")


if __name__ == "__main__":
    main()
