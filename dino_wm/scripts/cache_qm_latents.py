"""Cache frozen DINOv2 latents over FULL pusht_noise trajectories at the MODEL-STEP
grid, for training the QRL quasimetric cost-to-go head.

Why this (vs scripts/cache_latents.py, which only caches start+goal frames for the
bridge `g`): the quasimetric needs (z_t, z_{t+1}) transition pairs and (z_t, z_goal)
future pairs in the SAME step units the planner rolls out in. The DINO-WM dynamics
advances one MODEL step = `frameskip` env frames, so we encode each trajectory at
frame indices [0, frameskip, 2*frameskip, ...]. Consecutive cached latents are then
exactly one model-step apart (constant local cost r = -1), so a head trained on them
measures cost-to-go in model steps -- matching goal_H.

Encoding reproduces VWorldModel.encode_obs byte-for-byte:
    frame uint8 --(dataset default_transform 224: Resize,CenterCrop,Normalize.5)-->
    (3,224,224) --(encoder_transform: Resize((224//16)*patch_size=196))-->
    DinoV2Encoder.forward --> (196, 384) patch tokens.

Also caches the raw env state per model-step frame (sim/512 coords,
[ax,ay,bx,by,theta,vx,vy]); state[:,0:2] is the pusher xy used to build the
manipulator mask at train time (identical helper as the planner).

Output (per split) under <data_path>/qm_latents/<split>/:
    latents.pth       (Ntot, 196, 384) float16   -- concatenated model-step latents
    states.pth        (Ntot, state_dim) float32   -- raw states (sim coords)
    traj_starts.pth   (n_traj,) int64             -- start offset of each traj in latents
    traj_lengths.pth  (n_traj,) int64             -- #model-steps per traj
    meta.json         frameskip, counts, model-step length histogram

GPU run (vast.ai):
    cd dino_wm && source $WS/activate.sh
    python scripts/cache_qm_latents.py --splits train val
Mac smoke (cpu; needs a small pusht_noise-shaped folder + decord):
    .../dino_wm_dev/bin/python scripts/cache_qm_latents.py \
        --data_path data/pusht_noise_smoke --splits train --device cpu --n_rollout 4
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.img_transforms import default_transform
from datasets.pusht_dset import PushTDataset
from models.dino import DinoV2Encoder


def pick_device(arg):
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def encode_frames(frames, encoder, enc_resize, device, batch):
    """frames: (N,3,224,224) default_transform'd -> (N,196,384) float16 on cpu."""
    out = []
    for i in range(0, len(frames), batch):
        chunk = enc_resize(frames[i:i + batch].to(device))
        out.append(encoder.forward(chunk).half().cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",
                    default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_noise"))
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--frameskip", type=int, default=5,
                    help="MUST match the dynamics model's frameskip (train.yaml=5).")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_rollout", type=int, default=None,
                    help="cap #trajectories per split (None = all). Bounds cache size.")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device={device} frameskip={args.frameskip}")
    encoder = DinoV2Encoder(name="dinov2_vits14",
                            feature_key="x_norm_patchtokens").to(device).eval()
    enc_image_size = (args.img_size // 16) * encoder.patch_size  # -> 196
    enc_resize = transforms.Resize(enc_image_size)
    print(f"encoder input {enc_image_size}x{enc_image_size} -> 196 tokens x {encoder.emb_dim}")

    for split in args.splits:
        # pusht_noise stores train/ and val/ folders; PushTDataset appends nothing,
        # so point data_path at the split folder directly.
        split_path = Path(args.data_path) / split
        if not (split_path / "states.pth").exists():
            print(f"skip {split} (missing {split_path}/states.pth)")
            continue
        dset = PushTDataset(data_path=str(split_path), transform=default_transform(args.img_size),
                            n_rollout=args.n_rollout, normalize_action=True, with_velocity=True)
        n_traj = len(dset)
        lat_chunks, state_chunks, starts, lengths = [], [], [], []
        cursor = 0
        for i in range(n_traj):
            T = int(dset.get_seq_length(i))
            idxs = list(range(0, T, args.frameskip))   # model-step grid, phase 0
            if len(idxs) < 2:
                # need at least one transition; skip degenerate short trajs
                print(f"  traj {i}: only {len(idxs)} model-steps (T={T}); skipped")
                continue
            obs, _, state, _ = dset.get_frames(i, idxs)
            z = encode_frames(obs["visual"], encoder, enc_resize, device, args.batch)  # (L,196,384) f16
            lat_chunks.append(z)
            state_chunks.append(state.float())                                          # (L,state_dim)
            starts.append(cursor)
            lengths.append(len(idxs))
            cursor += len(idxs)
            if (i + 1) % 50 == 0 or i == n_traj - 1:
                print(f"  [{split}] {i+1}/{n_traj} trajs, {cursor} model-steps cached")

        latents = torch.cat(lat_chunks, dim=0)            # (Ntot,196,384) f16
        states = torch.cat(state_chunks, dim=0)           # (Ntot,state_dim) f32
        starts = torch.tensor(starts, dtype=torch.long)
        lengths = torch.tensor(lengths, dtype=torch.long)

        out_dir = Path(args.data_path) / "qm_latents" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(latents, out_dir / "latents.pth")
        torch.save(states, out_dir / "states.pth")
        torch.save(starts, out_dir / "traj_starts.pth")
        torch.save(lengths, out_dir / "traj_lengths.pth")
        Lh = lengths.numpy()
        meta = {
            "frameskip": args.frameskip, "n_traj": int(len(lengths)),
            "n_model_steps": int(latents.shape[0]),
            "model_step_len_min": int(Lh.min()), "model_step_len_max": int(Lh.max()),
            "model_step_len_mean": float(Lh.mean()),
            "model_step_len_p50": float(np.percentile(Lh, 50)),
            "model_step_len_p90": float(np.percentile(Lh, 90)),
            "state_dim": int(states.shape[1]), "latent_shape": list(latents.shape[1:]),
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        gb = latents.numel() * 2 / 1e9
        print(f"[{split}] cached {latents.shape} ({gb:.2f} GB f16) + states {tuple(states.shape)} -> {out_dir}")
        print(f"[{split}] model-step traj length: min={meta['model_step_len_min']} "
              f"p50={meta['model_step_len_p50']:.0f} p90={meta['model_step_len_p90']:.0f} "
              f"max={meta['model_step_len_max']}  <-- set phi OFFSET near p90/max")

    print("Done.")


if __name__ == "__main__":
    main()
