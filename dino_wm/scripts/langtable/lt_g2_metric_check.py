"""Metric-faithfulness diagnostic (diagnose-before-spend): does the latent metric track
"blocks reached the goal" along the TRUE oracle path, independent of any learned dynamics?

(A) From cache: masked latent distance-to-goal at every model-step of each real val trajectory.
    A faithful metric falls monotonically from the start toward ~0 at the end.
(B) Exact-final consistency: re-encode each trajectory's FINAL dot frame vs its effector-free
    goal_clean (SAME state, differ only by the dot) -> masked distance should be ~0. A large
    value = a render/encoding/dot-masking bug (NOT a data problem).

Run (dino_wm env): python lt_g2_metric_check.py --cache /workspace/lt_cache --traj_dir /workspace/lt_traj
"""
import argparse
import glob
import numpy as np
import torch
import torchvision.transforms.functional as TF

NP, GG = 196, 14


def w2tok(ee, half, cx, cy):
    col = (1 - (ee[1] - cy) / half) / 2; row = (1 - (ee[0] - cx) / half) / 2
    return int(min(max(row * GG, 0), GG - 1)) * GG + int(min(max(col * GG, 0), GG - 1))


def masked(zf, ee, zg, half, cx, cy):  # per-patch L2, drop the dot patch
    d = np.linalg.norm(zf.astype(np.float32) - zg.astype(np.float32), axis=-1)  # (196,)
    d[w2tok(ee, half, cx, cy)] = 0.0
    return d.sum() / (NP - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/lt_cache")
    ap.add_argument("--traj_dir", default="/workspace/lt_traj")
    ap.add_argument("--nB", type=int, default=20)
    args = ap.parse_args()
    va = dict(np.load(f"{args.cache}/val.npz", allow_pickle=True))
    half, ctr = float(va["half_extent"]), va["center"]; cx, cy = float(ctr[0]), float(ctr[1])

    # ---- (A) true-path distance from cached latents (no dynamics) ----
    print("[A] latent distance-to-goal along the TRUE oracle path (cached, no dynamics):")
    starts, ends, mins, monos, frac_drop, near = [], [], [], [], 0, 0
    for ep in range(len(va["seq_lengths"])):
        S = int(va["seq_lengths"][ep])
        zg = va["goal_visual"][ep]
        d = np.array([masked(va["visual"][ep, s], va["proprio"][ep, s], zg, half, cx, cy) for s in range(S)])
        starts.append(d[0]); ends.append(d[-1]); mins.append(d.min())
        monos.append(np.mean(np.diff(d) < 0))   # frac of steps that decrease
        frac_drop += int(d[-1] < d[0]); near += int(d[-1] < 0.4 * d[0])
    starts, ends, mins = map(np.array, (starts, ends, mins))
    print(f"  n={len(starts)}  start={starts.mean():.2f}  final-grid-step={ends.mean():.2f}  "
          f"min-along-path={mins.mean():.2f}  (ratio final/start={ends.mean()/starts.mean():.2f})")
    print(f"  final<start: {frac_drop}/{len(starts)}   final<0.4*start: {near}/{len(starts)}   "
          f"mean monotone-decreasing fraction={np.mean(monos):.2f}")

    # ---- (B) exact-final consistency: re-encode final dot frame vs goal_clean ----
    print("\n[B] exact-final-frame vs goal consistency (re-encode, SAME state, dot-masked -> want ~0):")
    base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().cuda()

    def enc(imgs):
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
        x = TF.normalize(x, [0.5] * 3, [0.5] * 3)
        x = TF.resize(x, [196, 196], antialias=True).cuda()
        with torch.no_grad():
            return base.forward_features(x)["x_norm_patchtokens"].cpu().numpy()
    files = sorted(glob.glob(f"{args.traj_dir}/w*_t*.npz"))[:args.nB]
    consist, raw = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        zf, zg = enc([d["frames"][-1], d["goal_clean"]])
        ee = np.asarray(d["proprio"][-1])[:2]
        consist.append(masked(zf, ee, zg, half, cx, cy))
        raw.append(np.linalg.norm(zf.astype(np.float32) - zg.astype(np.float32), axis=-1).mean())
    print(f"  n={len(consist)}  final-vs-goal masked dist (dot removed)={np.mean(consist):.3f}  "
          f"(unmasked={np.mean(raw):.3f})  -- compare to start dist ~{starts.mean():.1f}")
    print(f"\n[VERDICT] metric faithful if (A) final/start ratio is small AND (B) masked ~0. "
          f"Large (B) => render/encode/mask bug; flat (A) => metric doesn't track the goal.")


if __name__ == "__main__":
    main()
