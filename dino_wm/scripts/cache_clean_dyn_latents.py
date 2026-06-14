"""Step A of the clean-scene dynamics retrain: cache green-T-FREE dynamics latents.

Renders each pusht_noise trajectory CLEAN (with_target=False) from its states at the
model-step grid [0, fs, 2fs, ...], encodes with the frozen DINOv2 EXACTLY as
VWorldModel.encode_obs / cache_dynamics_latents (uint8 -> /255 -> default_transform(224)
-> Resize(196) -> DinoV2Encoder), and stores per-trajectory latents. Unlike
cache_dynamics_latents (which reads the GREEN-T mp4s), this re-renders clean frames,
and unlike multicolor (fixed length) it handles pusht_noise's VARIABLE length by
padding to max model-steps and storing n_steps (the seq-length-aware slicer uses it).

Output under <data_path>/dyn_latents_clean/<split>/:
    visual.pth   (n, maxS, 196, 384) float16  -- CLEAN model-step visual latents (padded)
    proprio.pth  (n, maxS, pdim) float32       -- NORMALIZED proprio (padded)
    actions.pth  (n, maxT, adim) float32       -- NORMALIZED per-env-frame actions (padded)
    states.pth   (n, maxS, sdim) float32       -- raw grid states (padded; diagnostics)
    n_steps.pth  (n,) int64                     -- valid model-steps per traj (S_i)
    meta.json    frameskip, cache_stride, maxS, maxT, n_traj, proprio_dim, action_dim, clean=True

  DATASET_DIR=/workspace/data python scripts/cache_clean_dyn_latents.py \
    --data_path /workspace/data/pusht_noise --frameskip 5 --max_traj 4000 --batch 256
"""
import os, sys, json, argparse
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.img_transforms import default_transform
from datasets.pusht_dset import PushTDataset
from datasets.rigid_goal_render import make_env, render_state
from models.dino import DinoV2Encoder


@torch.no_grad()
def cache_split(dset, env, fs, encoder, enc_resize, tfm, device, batch):
    n = len(dset)
    seqs = [int(dset.get_seq_length(i)) for i in range(n)]
    grids = [list(range(0, L, fs)) for L in seqs]          # env-frame grid per traj
    n_steps = np.array([len(g) for g in grids], dtype=np.int64)
    maxS = int(n_steps.max())
    maxT = int(max(seqs))
    pdim, sdim, adim = dset.proprio_dim, dset.state_dim, dset.action_dim

    vis = torch.zeros(n, maxS, 196, encoder.emb_dim, dtype=torch.float16)
    prop = torch.zeros(n, maxS, pdim, dtype=torch.float32)
    states = torch.zeros(n, maxS, sdim, dtype=torch.float32)
    actions = torch.zeros(n, maxT, adim, dtype=torch.float32)

    buf, slots = [], []

    def flush():
        if not buf:
            return
        x = enc_resize(torch.stack(buf).to(device))
        z = encoder.forward(x).cpu().half()
        for (ti, si), zi in zip(slots, z):
            vis[ti, si] = zi
        buf.clear(); slots.clear()

    for i in range(n):
        L = seqs[i]
        st_all = dset.states[i, :L]                         # (L, sdim) raw (incl velocity)
        actions[i, :L] = dset.actions[i, :L]                # normalized env-frame actions
        for si, fidx in enumerate(grids[i]):
            s5 = st_all[fidx, :5].numpy()
            img, _ = render_state(env, s5)                  # clean 224x224x3 uint8
            x = torch.tensor(img).float().div_(255.0).permute(2, 0, 1)  # (3,224,224)
            x = tfm(x)                                      # Resize/CenterCrop/Normalize
            prop[i, si] = dset.proprios[i, fidx]
            states[i, si] = st_all[fidx]
            buf.append(x); slots.append((i, si))
            if len(buf) >= batch:
                flush()
        if (i + 1) % max(1, n // 20) == 0:
            print(f"  rendered+encoded {i + 1}/{n} trajs", flush=True)
    flush()
    seq_lengths = np.array(seqs, dtype=np.int64)            # env-frame L_i (windowing constraint)
    return vis, prop, actions, states, n_steps, seq_lengths, maxS, maxT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--max_traj", type=int, default=4000, help="cap TRAIN trajs (val full)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
    enc_resize = transforms.Resize((224 // 16) * encoder.patch_size)   # Resize(196)
    tfm = default_transform(224)
    env = make_env(with_target=False)                                  # CLEAN scene

    for split in args.splits:
        sp = Path(args.data_path) / split
        if not sp.exists():
            print(f"skip {split} (missing)"); continue
        n_roll = args.max_traj if split == "train" else None
        dset = PushTDataset(data_path=str(sp), transform=tfm, normalize_action=True,
                            with_velocity=True, n_rollout=n_roll)
        vis, prop, actions, states, n_steps, seq_lengths, maxS, maxT = cache_split(
            dset, env, args.frameskip, encoder, enc_resize, tfm, device, args.batch)
        out = Path(args.data_path) / "dyn_latents_clean" / split
        out.mkdir(parents=True, exist_ok=True)
        torch.save(vis, out / "visual.pth")
        torch.save(prop, out / "proprio.pth")
        torch.save(actions, out / "actions.pth")
        torch.save(states, out / "states.pth")
        torch.save(torch.tensor(n_steps), out / "n_steps.pth")
        torch.save(torch.tensor(seq_lengths), out / "seq_lengths.pth")
        json.dump({"frameskip": args.frameskip, "cache_stride": args.frameskip,
                   "maxS": maxS, "maxT": maxT, "n_traj": len(dset), "clean": True,
                   "proprio_dim": dset.proprio_dim, "action_dim": dset.action_dim,
                   "variable_length": True}, open(out / "meta.json", "w"))
        print(f"[{split}] visual {tuple(vis.shape)} n_steps[min/max {int(n_steps.min())}/{int(n_steps.max())}] -> {out}")
    print("Done.")


if __name__ == "__main__":
    main()
