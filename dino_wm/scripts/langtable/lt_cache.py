"""Encode the trajectory corpus -> cached latents for the G2 dynamics smoke (dino_wm env).

Per trajectory: subsample dot frames at the frameskip grid -> DINOv2 patch latents (faithful
recipe: Normalize(0.5), Resize(196), dinov2_vits14, x_norm_patchtokens -> (196,384)); proprio =
EE xy at grid; actions per MODEL-step = concat of `frameskip` env actions; states (block xy) at
grid; goal = encoded effector-free goal_clean. Splits by trajectory. Saves train.npz / val.npz.

Run (dino_wm env): python lt_cache.py --traj_dir /workspace/lt_traj --out_dir /workspace/lt_cache --frameskip 5
"""
import argparse
import glob
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF


def encoder(device):
    base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)

    def enc(frames, batch=64):
        outs = []
        with torch.no_grad():
            for i in range(0, len(frames), batch):
                x = torch.from_numpy(frames[i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
                x = TF.normalize(x, [0.5] * 3, [0.5] * 3)
                x = TF.resize(x, [196, 196], antialias=True).to(device)
                outs.append(base.forward_features(x)["x_norm_patchtokens"].cpu().numpy().astype(np.float16))
        return np.concatenate(outs, 0)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", default="/workspace/lt_traj")
    ap.add_argument("--out_dir", default="/workspace/lt_cache")
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fs = args.frameskip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = encoder(device)

    files = sorted(glob.glob(os.path.join(args.traj_dir, "w*_t*.npz")))
    print(f"{len(files)} trajectory files")

    per = []  # per-traj dict
    Smax = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        T = int(d["seq_length"])
        grid = list(range(0, T, fs))
        S = len(grid)
        vis = enc(d["frames"][grid])                       # (S,196,384) f16
        prop = d["proprio"][grid].astype(np.float32)       # (S,2)
        bxy = d["block_xy"][grid].astype(np.float32)       # (S,8,2)
        acts = d["actions"].astype(np.float32)             # (T,2)
        am = np.zeros((S, fs * 2), np.float32)             # per model-step action = fs env actions
        for s in range(S):
            chunk = acts[grid[s]: grid[s] + fs]
            am[s, : chunk.size] = chunk.reshape(-1)
        gvis = enc(d["goal_clean"][None])[0]               # (196,384) f16
        per.append(dict(vis=vis, prop=prop, bxy=bxy, am=am, S=S, gvis=gvis,
                        gxy=d["goal_xy"].astype(np.float32),
                        sb=str(d["start_block"]), tb=str(d["target_block"]),
                        success=bool(d["success"])))
        Smax = max(Smax, S)
    blocks = [b.decode() if isinstance(b, bytes) else str(b) for b in d["blocks"]]
    half = float(d["half_extent"]); center = d["center"].astype(np.float32); size = int(d["size"])

    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(per)); rng.shuffle(idx)
    n_val = max(1, int(len(per) * args.val_frac))
    splits = {"val": idx[:n_val], "train": idx[n_val:]}

    def pack(ids, path):
        n = len(ids)
        vis = np.zeros((n, Smax, 196, 384), np.float16)
        prop = np.zeros((n, Smax, 2), np.float32)
        am = np.zeros((n, Smax, fs * 2), np.float32)
        bxy = np.zeros((n, Smax, 8, 2), np.float32)
        seq = np.zeros(n, np.int64)
        gvis = np.zeros((n, 196, 384), np.float16)
        gxy = np.zeros((n, 8, 2), np.float32)
        sb, tb, succ = [], [], []
        for j, i in enumerate(ids):
            p = per[i]; S = p["S"]
            vis[j, :S] = p["vis"]; prop[j, :S] = p["prop"]; am[j, :S] = p["am"]; bxy[j, :S] = p["bxy"]
            seq[j] = S; gvis[j] = p["gvis"]; gxy[j] = p["gxy"]
            sb.append(p["sb"]); tb.append(p["tb"]); succ.append(p["success"])
        np.savez_compressed(path, visual=vis, proprio=prop, actions=am, block_xy=bxy,
                            seq_lengths=seq, goal_visual=gvis, goal_xy=gxy,
                            start_block=np.array(sb), target_block=np.array(tb), success=np.array(succ),
                            blocks=np.array(blocks), frameskip=np.int32(fs), half_extent=np.float32(half),
                            center=center, size=np.int32(size))
        print(f"  {path}: n={n} Smax={Smax}")

    for split, ids in splits.items():
        pack(ids, os.path.join(args.out_dir, f"{split}.npz"))
    print(f"DONE: {len(per)} trajs -> {args.out_dir} (frameskip={fs}, proprio_dim=2, action_dim=2)")


if __name__ == "__main__":
    main()
