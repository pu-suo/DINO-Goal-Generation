"""G1-closing slices on the NEW (dot) render. Runs in dino_wm env (torch+dinov2+sklearn).

  Baseline : per-patch identity probe on REG clean (re-measured on the corrected render).
  Slice A  : same-color/same-shape CONTACT configs -> per-category boundary confusion +
             pair-block position-within-radius (graceful-degradation crux).
  Slice B  : DOT frames -> identity/position with the DOT PATCH MASKED (apples-to-apples
             with the deployable pusher-masked energy) + contacted/nearest-block sub-check.
  Slice C  : numerical dot-position check (detected dot centroid vs projected EE) +
             per-shape/color systematic-error scan.
  Disp     : goal-pairs displacement -- instructed block moves most + toward anchor; drift.

Run: python lt_slices.py --reg "/workspace/g1parts/part*.npz" --contact /workspace/lt_contact.npz
"""
import argparse
import collections
import glob

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.linear_model import LogisticRegression

RADIUS = 0.05
GRID = 14


def load_glob(pattern, keys):
    files = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    acc = {k: [] for k in keys}
    off = 0
    meta = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in keys:
            acc[k].append(d[k])
        if "episode" in acc:
            acc["episode"][-1] = acc["episode"][-1] + off
            off += int(d["episode"].max()) + 1
        meta = {m: d[m] for m in ("blocks", "half_extent", "center", "size") if m in d.files}
    out = {k: np.concatenate(acc[k], 0) for k in keys}
    out.update(meta)
    return out


def encoder(device):
    base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)

    def enc(frames, batch=64):
        outs = []
        with torch.no_grad():
            for i in range(0, len(frames), batch):
                x = torch.from_numpy(frames[i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
                x = TF.normalize(x, [0.5] * 3, [0.5] * 3)
                x = TF.resize(x, [196, 196], antialias=True).to(device)
                outs.append(base.forward_features(x)["x_norm_patchtokens"].cpu().numpy())
        return np.concatenate(outs, 0)  # (N,196,384)
    return enc


def make_mapping(half, center, size):
    H, cx, cy, S = float(half), float(center[0]), float(center[1]), int(size)

    def w2patch(x, y):
        col = (1 - (y - cy) / H) / 2 * S
        row = (1 - (x - cx) / H) / 2 * S
        return int(np.clip(row / S * GRID, 0, GRID - 1)), int(np.clip(col / S * GRID, 0, GRID - 1))

    def patch2w(pr, pc):
        col, row = (pc + 0.5) / GRID, (pr + 0.5) / GRID
        return cx + H * (1 - 2 * row), cy + H * (1 - 2 * col)

    def w2pix(x, y):
        return (1 - (y - cy) / H) / 2 * S, (1 - (x - cx) / H) / 2 * S  # col,row
    return w2patch, patch2w, w2pix


def patch_labels(block_xy, block_mask, nblk, w2patch):
    N = len(block_xy)
    lab = np.full((N, GRID * GRID), nblk, np.int64)
    for n in range(N):
        for b in range(nblk):
            if block_mask[n, b] < 0.5:
                continue
            pr, pc = w2patch(*block_xy[n, b])
            lab[n, pr * GRID + pc] = b
    return lab


def decode_positions(predN, block_xy, block_mask, nblk, patch2w, exclude_tok=None):
    """For each (frame, block): centroid of patches predicted==b -> world; err vs truth.
    exclude_tok[n] = set of token indices to ignore (e.g. dot patch). Returns list of (err, block, found)."""
    res = []
    for i in range(predN.shape[0]):
        ex = exclude_tok[i] if exclude_tok is not None else set()
        for b in range(nblk):
            if block_mask[i, b] < 0.5:
                continue
            toks = [t for t in np.where(predN[i] == b)[0] if t not in ex]
            if not toks:
                res.append((None, b, i))
                continue
            pw = np.mean([patch2w(t // GRID, t % GRID) for t in toks], 0)
            res.append((float(np.linalg.norm(pw - block_xy[i, b])), b, i))
    return res


def within(res):
    e = [r[0] for r in res if r[0] is not None]
    found = len(e) / max(1, len(res))
    e = np.array(e)
    return (e < RADIUS).mean() if len(e) else 0.0, np.median(e) if len(e) else 9.9, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", default="/workspace/g1parts/part*.npz")
    ap.add_argument("--contact", default="/workspace/lt_contact.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    reg = load_glob(args.reg, ["clean", "dot", "ee", "block_xy", "block_mask", "kind",
                               "episode", "instruction", "start_block", "target_block"])
    con = load_glob(args.contact, ["clean", "block_xy", "block_mask", "category", "pair_a", "pair_b"])
    blocks = [b.decode() if isinstance(b, bytes) else str(b) for b in reg["blocks"]]
    nblk = len(blocks)
    w2patch, patch2w, w2pix = make_mapping(reg["half_extent"], reg["center"], reg["size"])
    S = int(reg["size"])
    enc = encoder(device)
    print(f"REG {len(reg['clean'])} frames, CON {len(con['clean'])} configs, device={device}")

    # split REG by episode
    rng = np.random.RandomState(args.seed)
    uep = np.unique(reg["episode"]); rng.shuffle(uep)
    te_ep = set(uep[:max(1, int(0.3 * len(uep)))].tolist())
    te = np.array([e in te_ep for e in reg["episode"]]); tr = ~te

    # train per-patch probe on REG CLEAN train frames
    Fclean = enc(reg["clean"])
    lab = patch_labels(reg["block_xy"], reg["block_mask"], nblk, w2patch)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             multi_class="multinomial", n_jobs=-1).fit(
        Fclean[tr].reshape(-1, 384), lab[tr].reshape(-1))

    def eval_id(feat, labels, sel):
        pred = clf.predict(feat[sel].reshape(-1, 384)).reshape(sel.sum(), GRID * GRID)
        y = labels[sel]
        blk = y != nblk
        idacc = (pred.reshape(-1)[blk.reshape(-1)] == y.reshape(-1)[blk.reshape(-1)]).mean()
        return pred, idacc

    # ---------- BASELINE (REG clean test) ----------
    predB, idB = eval_id(Fclean, lab, te)
    resB = decode_positions(predB, reg["block_xy"][te], reg["block_mask"][te], nblk, patch2w)
    wB = within(resB)
    print(f"\n[BASELINE clean/new-render] id={idB:.3f}  pos<0.05u={wB[0]:.3f} median={wB[1]:.4f}u found={wB[2]:.3f}")

    # ---------- SLICE A (contact) ----------
    Fcon = enc(con["clean"])
    labC = patch_labels(con["block_xy"], con["block_mask"], nblk, w2patch)
    allc = np.ones(len(con["clean"]), bool)
    predC, _ = eval_id(Fcon, labC, allc)
    print("\n[SLICE A contact] per-category pair-block position-within-radius:")
    cats = con["category"]
    for cat in sorted(set(cats.tolist())):
        m = cats == cat
        pa, pb = con["pair_a"][m], con["pair_b"][m]
        # pair blocks only
        pr = predC[m]; bxy = con["block_xy"][m]; bmk = con["block_mask"][m]
        errs, bconf = [], []
        for i in range(m.sum()):
            for b in (pa[i], pb[i]):
                toks = np.where(pr[i] == b)[0]
                if len(toks):
                    pw = np.mean([patch2w(t // GRID, t % GRID) for t in toks], 0)
                    errs.append(np.linalg.norm(pw - bxy[i, b]))
                # boundary confusion: is the pair-block's true patch predicted as its twin?
                pp = w2patch(*bxy[i, b]); tok = pp[0] * GRID + pp[1]
                twin = pb[i] if b == pa[i] else pa[i]
                bconf.append(int(pr[i, tok] == twin))
        errs = np.array(errs)
        wr = (errs < RADIUS).mean() if len(errs) else 0.0
        print(f"  {cat:12s} pos<0.05u={wr:.2f} median={np.median(errs):.4f}u  twin-confusion={np.mean(bconf):.2f} (n_pairs={m.sum()})")

    # ---------- SLICE B (dot frames, dot patch masked) ----------
    Fdot = enc(reg["dot"])
    dotsel = (reg["kind"] == "rollout") | (reg["kind"] == "start")
    sel = dotsel & te
    # dot token per frame from ee
    ex = {}
    for i in np.where(sel)[0]:
        pr_, pc_ = w2patch(*reg["ee"][i])
        ex[i] = {pr_ * GRID + pc_}
    predD = clf.predict(Fdot[sel].reshape(-1, 384)).reshape(sel.sum(), GRID * GRID)
    # identity excluding dot patch
    labD = patch_labels(reg["block_xy"], reg["block_mask"], nblk, w2patch)[sel]
    idx_local = {gi: li for li, gi in enumerate(np.where(sel)[0])}
    ex_local = [set() for _ in range(sel.sum())]
    for gi, toks in ex.items():
        ex_local[idx_local[gi]] = toks
    blkmask = labD != nblk
    for li in range(sel.sum()):
        for t in ex_local[li]:
            blkmask[li, t] = False  # don't count dot-occupied patch
    idD = (predD[blkmask] == labD[blkmask]).mean()
    resD = decode_positions(predD, reg["block_xy"][sel], reg["block_mask"][sel], nblk, patch2w,
                            exclude_tok=ex_local)
    wD = within(resD)
    print(f"\n[SLICE B dot/masked] id={idD:.3f}  pos<0.05u={wD[0]:.3f} median={wD[1]:.4f}u found={wD[2]:.3f}")
    # contacted/nearest block sub-check
    near_res = []
    selidx = np.where(sel)[0]
    for li, gi in enumerate(selidx):
        dists = np.linalg.norm(reg["block_xy"][gi] - reg["ee"][gi], axis=1)
        dists[reg["block_mask"][gi] < 0.5] = 9.9
        b = int(dists.argmin())
        toks = [t for t in np.where(predD[li] == b)[0] if t not in ex_local[li]]
        if toks:
            pw = np.mean([patch2w(t // GRID, t % GRID) for t in toks], 0)
            near_res.append((float(np.linalg.norm(pw - reg["block_xy"][gi, b])), b, li))
        else:
            near_res.append((None, b, li))
    wN = within(near_res)
    print(f"  [contacted/nearest block] pos<0.05u={wN[0]:.3f} median={wN[1]:.4f}u found={wN[2]:.3f} (n={len(near_res)})")

    # ---------- SLICE C: dot-position numerical check ----------
    errs_dot = []
    for i in np.where(dotsel)[0][:200]:
        img = reg["dot"][i]
        white = ((img[:, :, 0] > 230) & (img[:, :, 1] > 230) & (img[:, :, 2] > 230))
        ys, xs = np.where(white)
        if len(xs) < 2:
            continue
        cx_d, cy_d = xs.mean(), ys.mean()
        col, row = w2pix(*reg["ee"][i])
        errs_dot.append(np.hypot(cx_d - col, cy_d - row))
    errs_dot = np.array(errs_dot)
    print(f"\n[SLICE C dot-position] detected-dot vs projected-EE: mean={errs_dot.mean():.2f}px "
          f"median={np.median(errs_dot):.2f}px max={errs_dot.max():.2f}px (n={len(errs_dot)})")
    # systematic per-shape / per-color baseline error
    pbe = collections.defaultdict(list)
    for err, b, i in resB:
        if err is not None:
            pbe[blocks[b]].append(err)
    print("[SLICE C systematic] per-block baseline median err (u):")
    print("   " + "  ".join(f"{b.split('_')[0][0]}{b.split('_')[1][:2]}={np.median(v):.3f}" for b, v in pbe.items()))

    # ---------- Displacement check (goal pairs) ----------
    print("\n[DISPLACEMENT goal-pairs] instructed-block-moves-most + toward-anchor:")
    eps = reg["episode"]
    ok_most, ok_toward, drifts, n = 0, 0, [], 0
    for e in np.unique(eps):
        si = np.where((eps == e) & (reg["kind"] == "start"))[0]
        gi = np.where((eps == e) & (reg["kind"] == "goal"))[0]
        if not len(si) or not len(gi):
            continue
        si, gi = si[0], gi[0]
        sb = str(reg["start_block"][si]); tb = str(reg["target_block"][si])
        if sb not in blocks or tb not in blocks:
            continue
        sbi, tbi = blocks.index(sb), blocks.index(tb)
        disp = np.linalg.norm(reg["block_xy"][gi] - reg["block_xy"][si], axis=1)
        moved_most = int(disp.argmax() == sbi)
        # toward anchor: did start_block get closer to target_block?
        d0 = np.linalg.norm(reg["block_xy"][si, sbi] - reg["block_xy"][si, tbi])
        d1 = np.linalg.norm(reg["block_xy"][gi, sbi] - reg["block_xy"][gi, tbi])
        toward = int(d1 < d0)
        other = np.delete(disp, sbi)
        ok_most += moved_most; ok_toward += toward; drifts.append(other.mean()); n += 1
    if n:
        print(f"  n={n}  instructed-moved-most={ok_most/n:.2f}  moved-toward-anchor={ok_toward/n:.2f}  "
              f"mean-other-block-drift={np.mean(drifts):.4f}u")


if __name__ == "__main__":
    main()
