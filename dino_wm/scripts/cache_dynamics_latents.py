"""Pre-encode multicolor trajectory latents at the MODEL-STEP grid for fast dynamics
training (skips the per-step mp4 decode + frozen-DINOv2 forward that makes train.py
dataloader-bound at ~16h/epoch on 10k trajs).

The dynamics predictor advances ONE model step = `frameskip` env frames, and the CEM
planner rolls out on that same grid. So we encode each trajectory at env-frame indices
[0, fs, 2fs, ...] and train on windows of consecutive model-steps. This is a (slightly
less phase-augmented) but deployment-MATCHED sample of the same one-step prediction
problem -- the physics relation z_t -> z_{t+fs} is phase-independent.

Encoding reproduces VWorldModel.encode_obs byte-for-byte (the cached visual ==
encode_obs()['visual']; verify with --selfcheck):
    frame uint8 -> /255 -> default_transform(224) -> Resize((224//16)*patch=196)
                -> DinoV2Encoder.forward -> (196, 384)

Output under <data_path>/dyn_latents/<split>/:
    visual.pth   (n_traj, S, 196, 384) float16  -- model-step visual latents
    proprio.pth  (n_traj, S, proprio_dim) float32 -- NORMALIZED proprio (dataset units)
    actions.pth  (n_traj, T, action_dim) float32  -- per-ENV-frame normalized actions
                 (the slicer concatenates fs of them per model-step, exactly as train.py)
    states.pth   (n_traj, S, state_dim) float32   -- raw states at the grid (diagnostics)
    meta.json    frameskip, S, proprio_dim, action_dim, n_traj

Run (box):
    python scripts/cache_dynamics_latents.py --data_path $DATASET_DIR/pusht_multicolor_10k \
        --frameskip 5 --batch 256 --selfcheck
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.img_transforms import default_transform
from datasets.pusht_multicolor_dset import PushTMultiColorDataset
from models.dino import DinoV2Encoder


@torch.no_grad()
def encode_split(dset, fs, encoder, enc_resize, device, batch):
    n = len(dset)
    T = dset.get_seq_length(0)
    grid = list(range(0, T, fs))          # model-step env-frame indices
    S = len(grid)
    vis = torch.empty(n, S, 196, encoder.emb_dim, dtype=torch.float16)
    prop = torch.empty(n, S, dset.proprio_dim, dtype=torch.float32)
    states = torch.empty(n, S, dset.state_dim, dtype=torch.float32)
    actions = torch.empty(n, T, dset.action_dim, dtype=torch.float32)

    buf, slots = [], []   # (traj_idx, step_idx) for each buffered frame

    def flush():
        if not buf:
            return
        x = enc_resize(torch.stack(buf).to(device))
        z = encoder.forward(x).cpu().half()
        for (ti, si), zi in zip(slots, z):
            vis[ti, si] = zi
        buf.clear(); slots.clear()

    for i in range(n):
        obs, act, state, _ = dset.get_frames(i, list(range(T)))
        actions[i] = act                                  # all env-frame actions
        for si, fidx in enumerate(grid):
            prop[i, si] = obs["proprio"][fidx]
            states[i, si] = state[fidx]
            buf.append(obs["visual"][fidx]); slots.append((i, si))
            if len(buf) >= batch:
                flush()
        if (i + 1) % max(1, n // 10) == 0:
            print(f"  encoded {i + 1}/{n}")
    flush()
    return vis, prop, actions, states, grid, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--selfcheck", action="store_true",
                    help="verify cached frame-0 latents match the standing start_latents cache")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
    enc_resize = transforms.Resize((224 // 16) * encoder.patch_size)
    tfm = default_transform(224)

    for split in args.splits:
        sp = Path(args.data_path) / split
        if not sp.exists():
            print(f"skip {split} (missing)"); continue
        dset = PushTMultiColorDataset(data_path=str(sp), transform=tfm,
                                      normalize_action=True, with_velocity=True)
        vis, prop, actions, states, grid, S = encode_split(
            dset, args.frameskip, encoder, enc_resize, device, args.batch)
        out = Path(args.data_path) / "dyn_latents" / split
        out.mkdir(parents=True, exist_ok=True)
        torch.save(vis, out / "visual.pth")
        torch.save(prop, out / "proprio.pth")
        torch.save(actions, out / "actions.pth")
        torch.save(states, out / "states.pth")
        json.dump({"frameskip": args.frameskip, "S": S, "n_traj": len(dset),
                   "proprio_dim": dset.proprio_dim, "action_dim": dset.action_dim,
                   "grid": grid}, open(out / "meta.json", "w"))
        print(f"[{split}] visual {tuple(vis.shape)} proprio {tuple(prop.shape)} -> {out}")

        if args.selfcheck:
            latents_dir = Path(args.data_path) / "latents" / split / "start_latents.pth"
            if latents_dir.exists():
                cached = torch.load(latents_dir).float()
                cos = torch.nn.functional.cosine_similarity(
                    vis[:, 0].float().reshape(len(vis), -1),
                    cached.reshape(len(cached), -1), dim=-1)
                print(f"  selfcheck cos(dyn step0, start_latents): mean {cos.mean():.4f} "
                      f"min {cos.min():.4f}  (expect ~1.0)")
    print("Done.")


if __name__ == "__main__":
    main()
