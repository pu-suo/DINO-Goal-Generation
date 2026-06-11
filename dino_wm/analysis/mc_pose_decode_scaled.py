"""Like-for-like multicolor pose-decode probe: fit the linear (dual-ridge) pose decoder on
~16k ARBITRARY multicolor frames -- matching the single-T probe's protocol (n_fit=16000,
trajectory frames, masked at the frame's pusher) -- instead of the 2k one-frame-per-episode
fits that left the sample-vs-representation question open (mc_decode_followup.py: slope
still falling at n=2000; start frames no better than goal frames).

Verdict rule:
  ~5-10px at n=16k  -> multicolor pose decodes like single-T; the earlier 32px was fit-sample
                       starvation; NO representation blocker.
  plateau ~25-30px  -> true linear-separability deficit on multicolor (4 T-outline distractors)
                       -> independent Phase-1 risk flag (though CEM plans on latent-L2, not on a
                       linear pose readout -- indirect implication, flag not gate).

Encodes frames through the exact training pipeline (default_transform(224) -> Resize(196) ->
DinoV2Encoder x_norm_patchtokens), reproducing scripts/cache_latents.py.

Run (box, AFTER the GPU is free -- the 16k fit wants ~6GB VRAM):
  python analysis/mc_pose_decode_scaled.py --data_path $DATASET_DIR/pusht_multicolor \
    --latent_dir $DATASET_DIR/pusht_multicolor/latents
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.img_transforms import default_transform
from datasets.pusht_multicolor_dset import PushTMultiColorDataset, PushTMultiColorLatentGoalDataset
from analysis.pose_decode_probe import fit_linear, wrapped_deg
from analysis.fit_multicolor_pose_decoder import masked_flat, pose4, pusher_xy, goal_pose
from analysis.mc_pose_decode_followup import start_pose, apply_dec, metrics, show
from models.dino import DinoV2Encoder

POS_TOL_PX, ANG_TOL_DEG = 20.0, 20.0


@torch.no_grad()
def encode_split(dset, per_ep, encoder, enc_resize, device, batch, rng):
    """Sample per_ep frames/episode; encode through the frozen pipeline.
    Returns (latents (M,196,384) cpu fp32, block_pose (M,3), pusher_xy (M,2))."""
    zs, poses, pushers = [], [], []
    buf, buf_meta = [], []

    def flush():
        if not buf:
            return
        x = enc_resize(torch.stack(buf).to(device))
        zs.append(encoder.forward(x).cpu())
        buf.clear()

    for i in range(len(dset)):
        T = dset.get_seq_length(i)
        fr = np.sort(rng.choice(T, size=min(per_ep, T), replace=False))
        obs, _, state, _ = dset.get_frames(i, list(fr))
        st = state.numpy()
        for j in range(len(fr)):
            buf.append(obs["visual"][j])
            poses.append(st[j, 2:5])
            pushers.append(st[j, :2])
            if len(buf) >= batch:
                flush()
    flush()
    return torch.cat(zs), torch.tensor(np.array(poses), dtype=torch.float32), np.array(pushers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--per_ep_train", type=int, default=8)   # 2000 eps x 8 = 16k fit frames
    ap.add_argument("--per_ep_test", type=int, default=5)    # 400 eps x 5 = 2k eval frames
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--dilation", type=int, default=0)
    ap.add_argument("--fit_device", default="cuda")
    ap.add_argument("--out", default="analysis_outputs/pose_decode_probe/mc_decode_scaled.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
    enc_image_size = (224 // 16) * encoder.patch_size
    enc_resize = transforms.Resize(enc_image_size)
    tfm = default_transform(224)
    rng = np.random.RandomState(0)

    print("== encoding sampled frames (train fit set / test frame-eval set) ==")
    tr = PushTMultiColorDataset(data_path=os.path.join(args.data_path, "train"),
                                transform=tfm, normalize_action=False, with_velocity=True)
    te = PushTMultiColorDataset(data_path=os.path.join(args.data_path, "test"),
                                transform=tfm, normalize_action=False, with_velocity=True)
    Ztr, Ptr, PUtr = encode_split(tr, args.per_ep_train, encoder, enc_resize, device, args.batch, rng)
    Zte, Pte, PUte = encode_split(te, args.per_ep_test, encoder, enc_resize, device, args.batch, rng)
    print(f"encoded: train {tuple(Ztr.shape)}  test {tuple(Zte.shape)}")
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    Xtr, Xte = masked_flat(Ztr, PUtr, args.dilation), masked_flat(Zte, PUte, args.dilation)
    del Ztr, Zte
    Ytr, Yte = pose4(Ptr), pose4(Pte)

    # the standing test-split goal/start latents (the quantities g/planning care about)
    lg = PushTMultiColorLatentGoalDataset(args.latent_dir, args.data_path, "test")
    pu = pusher_xy(lg)
    Ggoal, Gstart = masked_flat(lg.goal, pu, args.dilation), masked_flat(lg.start, pu, args.dilation)
    Ygoal, Ystart = pose4(goal_pose(lg)), pose4(start_pose(lg))

    out = {}
    order = np.random.RandomState(1).permutation(len(Xtr))
    for n in (2000, 4000, 8000, 16000):
        if n > len(Xtr):
            break
        sub = order[:n]
        _, dec = fit_linear(Xtr[sub], Ytr[sub], Xte[:1], args.ridge_lambda, args.fit_device)
        out[f"frames{n}_test_frames"] = metrics(apply_dec(dec, Xte), Yte)
        out[f"frames{n}_test_goals"] = metrics(apply_dec(dec, Ggoal), Ygoal)
        out[f"frames{n}_test_starts"] = metrics(apply_dec(dec, Gstart), Ystart)
        print(f"-- n_fit={n} --")
        show("-> 2k held-out test FRAMES", out[f"frames{n}_test_frames"])
        show("-> 400 test GOAL latents", out[f"frames{n}_test_goals"])
        show("-> 400 test START latents", out[f"frames{n}_test_starts"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
