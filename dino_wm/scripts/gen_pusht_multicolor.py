"""
Phase 0.2 - generate the multi-color PushT dataset (CPU-bound, parallel).

Produces a layout that is drop-in compatible with the upstream PushT loader
(states/actions/velocities/seq_lengths/shapes + obses/*.mp4) PLUS the multi-color
label fields, a per-episode goal frame (block relocated to the NAMED target), and
a FROZEN color-location split:

    <out>/pusht_multicolor/
      train/   val/        # episodes whose targets all use TRAIN combos (dynamics + g training)
      test/                # episodes whose ACTIVE target uses a HELD-OUT (color,bin) combo
      split_manifest.json  # combo sets + params + counts (the frozen split)
      stats.pth            # action/state/proprio mean+std computed from train

    each split dir:
      states.pth (N,T,5)  rel_actions.pth (N,T,2)  abs_actions.pth (N,T,2)
      velocities.pth (N,T,2)  seq_lengths.pkl  shapes.pkl
      obses/episode_XXXXXX.mp4         goal_obses/episode_XXXXXX.png
      labels.pkl  (list of per-episode layout+label dicts)

Run (small smoke test, Mac dev env):
    cd dino_wm
    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python scripts/gen_pusht_multicolor.py \
        --out data/pusht_multicolor_smoke --n_train 6 --n_val 2 --n_test 4 --T 12 --workers 4

Real run (vast.ai): bump counts (e.g. --n_train 2000 --n_val 200 --n_test 400 --T 100 --workers <vcpus>).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame in every worker

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import imageio

# allow `python scripts/gen_pusht_multicolor.py` from the repo root
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from env.pusht.pusht_multicolor_env import PushTMultiColorEnv
from env.pusht import multicolor_sampler as mcs


# --- exploration policy (state-dependent; ensures block contact + diversity) ---
def policy_step(state, agent_xy, block_xy, rng, prev, mode):
    if mode == "brownian":
        return np.clip(0.8 * prev + rng.normal(0, 0.5, 2), -1, 1)
    # 'push_noise' (default): bias the pusher toward the block, with noise, and
    # occasional fully-random actions, so the block actually moves around.
    d = np.asarray(block_xy) - np.asarray(agent_xy)
    n = np.linalg.norm(d) + 1e-6
    a = 0.6 * (d / n) + rng.normal(0, 0.6, 2)
    if rng.random() < 0.2:
        a = rng.uniform(-1, 1, 2)
    return np.clip(a, -1, 1)


# --- one episode (runs in a worker process) -----------------------------------
def generate_one(task):
    (idx, layout, env_kwargs, T, policy, render_size, fps,
     split_dir) = task
    env = PushTMultiColorEnv(render_size=render_size, **env_kwargs)
    env.set_layout(layout)
    env.seed(int(layout["seed"]))
    env.reset_to_state = np.asarray(layout["init_state"]).copy()
    obs, state = env.reset()
    action_scale = env.action_scale

    rng = np.random.RandomState(int(layout["seed"]) + 777)
    frames = [obs["visual"]]
    states = [state]
    rel_acts, abs_acts = [], []
    prev = np.zeros(2)
    for t in range(T):
        agent_xy = state[:2]
        block_xy = state[2:4]
        a = policy_step(state, agent_xy, block_xy, rng, prev, policy)
        prev = a
        obs, _, _, info = env.step(a)
        ns = info["state"]
        rel_acts.append(a * action_scale)                 # pixel delta (loader divides by scale)
        abs_acts.append(agent_xy + a * action_scale)       # absolute target pixel
        frames.append(obs["visual"])
        states.append(ns)
        state = ns

    # align to length T: states/frames[0..T-1] with actions[0..T-1]
    states = np.asarray(states[:T], dtype=np.float32)
    frames = frames[:T]
    rel_acts = np.asarray(rel_acts[:T], dtype=np.float32)
    abs_acts = np.asarray(abs_acts[:T], dtype=np.float32)

    # goal frame: block at the named target, agent left at its start
    goal_state = np.asarray(layout["init_state"]).copy()
    goal_state[2:5] = np.asarray(layout["goal_pose"])
    env.reset_to_state = goal_state
    gobs, _ = env.reset()
    goal_frame = gobs["visual"]

    # write video + goal frame. Pin libx264 to ONE thread per encode: with many
    # parallel workers, libx264's default (~one thread/core) x workers explodes
    # the thread count and the kernel refuses new threads (EAGAIN -> empty mp4).
    vid_path = os.path.join(split_dir, "obses", f"episode_{idx:06d}.mp4")
    with imageio.get_writer(vid_path, fps=fps, macro_block_size=1, codec="libx264",
                            output_params=["-threads", "1"]) as w:
        for f in frames:
            w.append_data(f)
    imageio.imwrite(os.path.join(split_dir, "goal_obses", f"episode_{idx:06d}.png"), goal_frame)

    label = {
        "active_color": layout["active_color"],
        "active_idx": int(layout["active_idx"]),
        "active_bin": int(layout["targets"][layout["active_idx"]]["bin"]),
        "target_colors": [t["color"] for t in layout["targets"]],
        "target_poses": np.stack([np.asarray(t["pose"]) for t in layout["targets"]]).astype(np.float32),
        "target_bins": [int(t["bin"]) for t in layout["targets"]],
        "instruction": layout["instruction"],
        "template_id": int(layout["template_id"]),
        "goal_pose": np.asarray(layout["goal_pose"], dtype=np.float32),
        "init_state": np.asarray(layout["init_state"], dtype=np.float32),
        "seed": int(layout["seed"]),
    }
    return idx, states, rel_acts, abs_acts, label


# --- per-split driver ---------------------------------------------------------
def build_split(name, layouts, out_root, env_kwargs, T, policy, render_size, fps, workers):
    split_dir = os.path.join(out_root, name)
    os.makedirs(os.path.join(split_dir, "obses"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "goal_obses"), exist_ok=True)
    n = len(layouts)

    tasks = [
        (i, layouts[i], env_kwargs, T, policy, render_size, fps, split_dir)
        for i in range(n)
    ]
    results = [None] * n
    if workers <= 1:
        for task in tasks:
            r = generate_one(task)
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

    states = np.stack([r[1] for r in results])
    rel = np.stack([r[2] for r in results])
    abs_ = np.stack([r[3] for r in results])
    labels = [r[4] for r in results]
    velocities = states[:, :, 5:7] if states.shape[-1] >= 7 else np.zeros((n, T, 2), np.float32)
    states5 = states[:, :, :5]

    import torch
    torch.save(torch.from_numpy(states5), os.path.join(split_dir, "states.pth"))
    torch.save(torch.from_numpy(rel), os.path.join(split_dir, "rel_actions.pth"))
    torch.save(torch.from_numpy(abs_), os.path.join(split_dir, "abs_actions.pth"))
    torch.save(torch.from_numpy(velocities), os.path.join(split_dir, "velocities.pth"))
    with open(os.path.join(split_dir, "seq_lengths.pkl"), "wb") as f:
        pickle.dump([T] * n, f)
    with open(os.path.join(split_dir, "shapes.pkl"), "wb") as f:
        pickle.dump(["T"] * n, f)
    with open(os.path.join(split_dir, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)
    print(f"  [{name}] saved {n} episodes -> {split_dir}")
    return states5, rel, velocities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_multicolor"))
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=400, help="held-out color-location episodes")
    ap.add_argument("--T", type=int, default=100, help="raw steps per episode")
    ap.add_argument("--n_targets", type=int, default=4)
    ap.add_argument("--policy", choices=["push_noise", "brownian"], default="push_noise")
    ap.add_argument("--heldout_frac", type=float, default=0.2)
    ap.add_argument("--n_bins", type=int, default=3)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--workers", type=int,
                    default=min(12, max(1, (os.cpu_count() or 2) - 1)),
                    help="parallel episode workers; capped at 12 by default so the per-worker "
                         "single-threaded ffmpeg encoders don't exhaust kernel threads. Raise "
                         "cautiously if your box stays stable.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env_kwargs = dict(with_velocity=True, n_targets=args.n_targets, n_bins=args.n_bins)

    # frozen color-location split
    train_combos, test_combos = mcs.make_combo_split(
        n_targets=args.n_targets, n_bins=args.n_bins,
        heldout_frac=args.heldout_frac, seed=args.seed,
    )

    def make_layouts(count, base, allowed, active):
        return [
            mcs.sample_layout(
                base + i, n_targets=args.n_targets, with_velocity=True,
                n_bins=args.n_bins, allowed_combos=allowed, active_combos=active,
            )
            for i in range(count)
        ]

    print(f"Split: {len(train_combos)} train combos, {len(test_combos)} held-out combos")
    print("Sampling layouts...")
    train_layouts = make_layouts(args.n_train, 1_000_000, train_combos, None)
    val_layouts = make_layouts(args.n_val, 2_000_000, train_combos, None)
    test_layouts = make_layouts(args.n_test, 3_000_000, train_combos, test_combos)

    print("Generating splits...")
    s_tr, a_tr, v_tr = build_split("train", train_layouts, args.out, env_kwargs,
                                   args.T, args.policy, args.render_size, args.fps, args.workers)
    build_split("val", val_layouts, args.out, env_kwargs,
                args.T, args.policy, args.render_size, args.fps, args.workers)
    build_split("test", test_layouts, args.out, env_kwargs,
                args.T, args.policy, args.render_size, args.fps, args.workers)

    # stats from train split (for normalization when training a multicolor model)
    import torch
    rel_flat = a_tr.reshape(-1, 2) / 100.0  # to env action units (loader divides rel by action_scale)
    states_flat = s_tr.reshape(-1, 5)
    proprio_flat = np.concatenate([s_tr[:, :, :2], v_tr], axis=-1).reshape(-1, 4)
    stats = {
        "action_mean": torch.tensor(rel_flat.mean(0)), "action_std": torch.tensor(rel_flat.std(0) + 1e-6),
        "state_mean": torch.tensor(np.concatenate([states_flat.mean(0), v_tr.reshape(-1, 2).mean(0)])),
        "state_std": torch.tensor(np.concatenate([states_flat.std(0), v_tr.reshape(-1, 2).std(0)]) + 1e-6),
        "proprio_mean": torch.tensor(proprio_flat.mean(0)), "proprio_std": torch.tensor(proprio_flat.std(0) + 1e-6),
    }
    torch.save(stats, os.path.join(args.out, "stats.pth"))

    manifest = {
        "n_targets": args.n_targets, "n_bins": args.n_bins, "T": args.T,
        "heldout_frac": args.heldout_frac, "policy": args.policy, "seed": args.seed,
        "render_size": args.render_size,
        "counts": {"train": args.n_train, "val": args.n_val, "test": args.n_test},
        "train_combos": sorted([list(c) for c in train_combos]),
        "test_combos": sorted([list(c) for c in test_combos]),
    }
    with open(os.path.join(args.out, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Dataset at {args.out}")
    print(f"  split_manifest.json: {len(test_combos)} held-out combos frozen")


if __name__ == "__main__":
    main()
