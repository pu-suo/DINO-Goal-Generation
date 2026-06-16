"""G1: frozen-DINOv2 multi-object separability probe (the crux gate, plan §7/§11-risk-2).

Encodes top-down frames with frozen DINOv2 (faithful DINO-WM recipe: Normalize(0.5),
Resize(196), dinov2_vits14, x_norm_patchtokens -> 196x384), then fits a PCA->RidgeCV probe
from the patch features to per-block 2D position. Reports held-out (by EPISODE) per-block
position error vs the block2block success radius (0.05 world-u). Same-color blocks are the
separability stress. Prints TRAIN vs TEST error to separate underfit (weak features) from a
real ceiling, and an effector-position sanity probe (one salient object).

Run (dino_wm env w/ torch+dinov2+sklearn):
  python dino_wm/scripts/langtable/lt_g1_probe.py --npz "/workspace/g1parts/part*.npz" [--frames visible|hidden|both]
"""
import argparse
import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

RADIUS = 0.05  # block2block success radius (world units)


def load_npz_glob(pattern):
    files = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    assert files, f"no files match {pattern}"
    keys = ["visible", "hidden", "block_xy", "block_yaw", "block_mask", "kind", "instruction", "episode"]
    acc = {k: [] for k in keys}
    off = 0
    blocks = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        blocks = d["blocks"]
        for k in keys:
            acc[k].append(d[k])
        acc["episode"][-1] = acc["episode"][-1] + off
        off += int(d["episode"].max()) + 1
    out = {k: np.concatenate(acc[k], 0) for k in keys}
    out["blocks"] = blocks
    print(f"loaded {len(files)} file(s), {len(out['visible'])} frames, {off} episodes")
    return out


def encode(frames_uint8, device, batch=32):
    base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(frames_uint8), batch):
            x = torch.from_numpy(frames_uint8[i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
            x = TF.normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            x = TF.resize(x, [196, 196], antialias=True).to(device)
            f = base.forward_features(x)["x_norm_patchtokens"]  # (b,196,384)
            feats.append(f.reshape(f.shape[0], -1).cpu().numpy())
    X = np.concatenate(feats, 0).astype(np.float32)
    return X


def per_block_err(pred, true, nblocks):
    p = pred.reshape(-1, nblocks, 2)
    t = true.reshape(-1, nblocks, 2)
    return np.linalg.norm(p - t, axis=2)  # (N, nblocks)


def fit_probe(Xtr, Ytr, Xte, k=128):
    k = min(k, Xtr.shape[0] - 1)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    reg = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(Ztr, Ytr)
    evr = float(pca.explained_variance_ratio_.sum())
    return reg.predict(Ztr), reg.predict(Zte), float(reg.alpha_), evr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/workspace/g1parts/part*.npz")
    ap.add_argument("--frames", choices=["visible", "hidden", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--pca", type=int, default=128)
    args = ap.parse_args()

    d = load_npz_glob(args.npz)
    blocks = [b.decode() if isinstance(b, bytes) else str(b) for b in d["blocks"]]
    nblk = len(blocks)
    Y = d["block_xy"].reshape(len(d["block_xy"]), -1).astype(np.float32)
    ep = d["episode"]
    kind = np.array([k.decode() if isinstance(k, bytes) else str(k) for k in d["kind"]])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.RandomState(args.seed)
    uep = np.unique(ep)
    rng.shuffle(uep)
    n_te = max(1, int(len(uep) * args.test_frac))
    te_ep = set(uep[:n_te].tolist())
    te = np.array([e in te_ep for e in ep])
    tr = ~te
    print(f"frames={len(Y)} episodes={len(uep)} device={device} | train={tr.sum()} test={te.sum()} (by episode)")
    print(f"label spread (per-coord std): {Y.std(0).mean():.4f} u")

    modes = ["visible", "hidden"] if args.frames == "both" else [args.frames]
    for mode in modes:
        X = encode(d[mode], device)
        fvar = X.std(0).mean()
        ptr, pte, alpha, evr = fit_probe(X[tr], Y[tr], X[te], k=args.pca)
        err = per_block_err(pte, Y[te], nblk)
        terr = per_block_err(ptr, Y[tr], nblk)
        base = per_block_err(np.repeat(Y[tr].mean(0, keepdims=True), te.sum(), 0), Y[te], nblk)

        print(f"\n=== {mode.upper()}  (feat std={fvar:.3f}, PCA{args.pca} evr={evr:.2f}, alpha={alpha:.0e}) ===")
        print(f"  TRAIN err {terr.mean():.4f}u | TEST err {err.mean():.4f}u ({err.mean()/RADIUS:.2f}x radius) "
              f"| baseline {base.mean():.4f}u")
        print(f"  TEST within 0.05u: {(err<RADIUS).mean():.2f}  within 0.025u: {(err<RADIUS/2).mean():.2f}")
        for kk in ["start", "goal"]:
            sel = (kind[te] == kk)
            if sel.any():
                print(f"  [{kk:5s}] test err {err[sel].mean():.4f}u  within0.05 {(err[sel]<RADIUS).mean():.2f} (n={sel.sum()})")
        pbe = err.mean(0)
        print("  per-block test err (u): " + "  ".join(
            f"{b.split('_')[0][0]}{b.split('_')[1][:2]}={pbe[i]:.3f}" for i, b in enumerate(blocks)))


if __name__ == "__main__":
    main()
