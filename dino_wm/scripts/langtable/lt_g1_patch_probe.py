"""G1 (data-efficient): per-PATCH block-identity separability probe.

The right test of "do frozen DINOv2 patches carry multi-object identity?" Each of the 14x14
patch tokens is labeled with the block whose center falls in it (or background), giving
~196x more samples than a global position regression. A linear (logistic) probe maps the
384-d patch token -> {8 blocks, background}. We report:
  - block-patch identity accuracy (of patches that contain a block, how often correct id),
  - SAME-COLOR confusion (red_moon vs red_pentagon, ...) -- the crux stress,
  - block-vs-background detectability,
  - decoded position error (centroid of predicted-block patches vs truth).
Held out by EPISODE. Run in dino_wm env (torch+dinov2+sklearn).
"""
import argparse
import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.linear_model import LogisticRegression

RADIUS = 0.05
HALF, CX, CY, SIZE, GRID = 0.32, 0.375, 0.0, 224, 14


def world_to_patch(x, y):
    col = (1.0 - (y - CY) / HALF) / 2.0
    row = (1.0 - (x - CX) / HALF) / 2.0
    pc = int(np.clip(col * GRID, 0, GRID - 1))
    pr = int(np.clip(row * GRID, 0, GRID - 1))
    return pr, pc  # patch (row, col); token index = pr*GRID + pc


def patch_to_world(pr, pc):
    # patch center -> world (inverse of world_to_patch)
    col = (pc + 0.5) / GRID
    row = (pr + 0.5) / GRID
    y = CY + HALF * (1.0 - 2.0 * col)
    x = CX + HALF * (1.0 - 2.0 * row)
    return x, y


PATCH_U = 2.0 * HALF / GRID  # world size of one patch (~0.046u)


def load(pattern):
    files = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    keys = ["visible", "hidden", "block_xy", "block_mask", "kind", "episode"]
    acc = {k: [] for k in keys}
    off = 0
    blocks = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        blocks = [b.decode() if isinstance(b, bytes) else str(b) for b in d["blocks"]]
        for k in keys:
            acc[k].append(d[k])
        acc["episode"][-1] = acc["episode"][-1] + off
        off += int(d["episode"].max()) + 1
    out = {k: np.concatenate(acc[k], 0) for k in keys}
    out["blocks"] = blocks
    return out


def encode_patches(frames, device, batch=32):
    base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
            x = TF.normalize(x, [0.5] * 3, [0.5] * 3)
            x = TF.resize(x, [196, 196], antialias=True).to(device)
            f = base.forward_features(x)["x_norm_patchtokens"]  # (b,196,384)
            out.append(f.cpu().numpy())
    return np.concatenate(out, 0)  # (N,196,384)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/workspace/g1parts/part*.npz")
    ap.add_argument("--frames", choices=["visible", "hidden"], default="hidden")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load(args.npz)
    blocks = d["blocks"]
    nblk = len(blocks)
    BG = nblk
    ep = d["episode"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = encode_patches(d[args.frames], device)  # (N,196,384)
    N = feats.shape[0]
    print(f"{args.frames}: {N} frames, {len(np.unique(ep))} episodes, device={device}")

    # per-patch labels: block id where a block center lands, else BG
    labels = np.full((N, GRID * GRID), BG, np.int64)
    blk_tok = np.zeros((N, nblk), np.int64)  # token index of each block
    for n in range(N):
        for b in range(nblk):
            if d["block_mask"][n, b] < 0.5:
                blk_tok[n, b] = -1
                continue
            pr, pc = world_to_patch(*d["block_xy"][n, b])
            tok = pr * GRID + pc
            labels[n, tok] = b
            blk_tok[n, b] = tok

    # split by episode
    rng = np.random.RandomState(args.seed)
    uep = np.unique(ep); rng.shuffle(uep)
    te_ep = set(uep[:max(1, int(0.3 * len(uep)))].tolist())
    te = np.array([e in te_ep for e in ep]); tr = ~te

    Xtr = feats[tr].reshape(-1, 384); ytr = labels[tr].reshape(-1)
    Xte = feats[te].reshape(-1, 384); yte = labels[te].reshape(-1)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             multi_class="multinomial", n_jobs=-1).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    # block-patch identity accuracy (patches that ARE a block)
    blk = yte != BG
    id_acc = (pred[blk] == yte[blk]).mean()
    # block-vs-bg detection
    det = ((pred != BG) == (yte != BG)).mean()
    print(f"\n=== PER-PATCH IDENTITY ({args.frames}) ===")
    print(f"  block-patch identity acc: {id_acc:.3f}  (chance=1/{nblk}={1/nblk:.3f})")
    print(f"  block-vs-bg patch acc:    {det:.3f}")
    # per-block recall + same-color confusion
    yte2 = yte[blk]; pr2 = pred[blk]
    print("  per-block recall:")
    for b in range(nblk):
        sel = yte2 == b
        if sel.sum():
            rec = (pr2[sel] == b).mean()
            # most-confused other block
            wrong = pr2[sel][pr2[sel] != b]
            conf = ""
            if len(wrong):
                vals, cnts = np.unique(wrong, return_counts=True)
                j = vals[cnts.argmax()]
                cn = blocks[j] if j < nblk else "bg"
                conf = f"  ->{cn}({cnts.max()}/{sel.sum()})"
            print(f"    {blocks[b]:16s} recall={rec:.2f} (n={sel.sum()}){conf}")
    # decoded position: centroid of predicted-block patches vs true (test frames)
    predN = pred.reshape(te.sum(), GRID * GRID)
    bxy_te = d["block_xy"][te]
    mask_te = d["block_mask"][te]
    errs = []
    for i in range(predN.shape[0]):
        for b in range(nblk):
            if mask_te[i, b] < 0.5:
                continue
            toks = np.where(predN[i] == b)[0]
            if len(toks) == 0:
                continue  # missed (counted separately by recall)
            prc = np.array([patch_to_world(t // GRID, t % GRID) for t in toks])
            pxy = prc.mean(0)
            errs.append(np.linalg.norm(pxy - bxy_te[i, b]))
    errs = np.array(errs)
    print(f"  decoded position err: mean={errs.mean():.4f}u median={np.median(errs):.4f}u "
          f"(patch={PATCH_U:.3f}u) within0.05u={np.mean(errs<RADIUS):.2f} within0.025u={np.mean(errs<RADIUS/2):.2f}")

    # same-color pair confusion summary
    same = {}
    for b in range(nblk):
        c = blocks[b].split("_")[0]
        same.setdefault(c, []).append(b)
    print("  same-color cross-confusion (frac of A's block-patches predicted as its same-color twin):")
    for c, bs in same.items():
        if len(bs) == 2:
            a, z = bs
            sa = yte2 == a; sz = yte2 == z
            fa = (pr2[sa] == z).mean() if sa.sum() else 0
            fz = (pr2[sz] == a).mean() if sz.sum() else 0
            print(f"    {c}: {blocks[a]}<->{blocks[z]}  {fa:.2f} / {fz:.2f}")


if __name__ == "__main__":
    main()
