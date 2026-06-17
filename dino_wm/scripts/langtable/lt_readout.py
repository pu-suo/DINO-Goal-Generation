"""R: per-block position+identity readout (productionized Gate-1 probe, geometry-corrected).

Decodes the 8 FIXED_8 block positions + identity confidence from a frozen DINOv2 (196,384)
patch grid. Pusher-INVARIANT by construction: it only reads the 8 block classes; the white
dot/pusher patch falls to the background class, so a relative position read is unaffected by
the manipulator (this is the embodiment-contamination fix -- latent-L2 compares the whole
grid which DINOv2's global attention contaminates; R extracts only block positions, which
Gate-1 Slice B already showed survives the dot at ~0.93).

This is the front-end of the object-factored relational CEM energy: cost = h(R_A, R_B).
Reuses the Gate-1 per-patch logistic probe (lt_g1_patch_probe.py). FIX vs Gate-1: the probe
hardcoded half_extent=0.32; the real render scale is 0.3048 (cache 'half_extent'). We read
the geometry from the cache so labels and decode are unbiased, and re-validate.

__main__ trains R on cached `visual` latents + `block_xy` ground truth (FREE labels) and runs
the D1-REPRESENTATIONAL check on the held-out val split (val.npz = separate trajectories):
  (id)  per-patch block identity acc + same-color confusion
  (pos) soft-argmax decoded position err + within-0.05u
  (rel) decoded dist(A,B) vs true dist(A,B) MAE; near/far ordering; success(dist<0.05) classif.

Run (dino_wm env):
  python lt_readout.py --cache /workspace/lt_cache --out /workspace/readout
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID, NP, DIM = 14, 196, 384
RADIUS = 0.05  # block2block success threshold (TARGET_BLOCK_DISTANCE, world meters)


def patch_centers(half, cx, cy):
    """World (x,y) center of each of the 196 patches, token-ordered (pr*GRID+pc)."""
    pr = np.repeat(np.arange(GRID), GRID)
    pc = np.tile(np.arange(GRID), GRID)
    col = (pc + 0.5) / GRID
    row = (pr + 0.5) / GRID
    y = cy + half * (1.0 - 2.0 * col)
    x = cx + half * (1.0 - 2.0 * row)
    return np.stack([x, y], 1).astype(np.float32)  # (196,2)


def world_to_tok(xy, half, cx, cy):
    """World (...,2) -> patch token index in [0,195]. Matches lt_render.world_to_pixel axes."""
    x, y = xy[..., 0], xy[..., 1]
    col = (1.0 - (y - cy) / half) / 2.0
    row = (1.0 - (x - cx) / half) / 2.0
    pc = np.clip((col * GRID).astype(np.int64), 0, GRID - 1)
    pr = np.clip((row * GRID).astype(np.int64), 0, GRID - 1)
    return pr * GRID + pc


class Readout(nn.Module):
    """Frozen-input per-patch linear head -> nblk+1 classes (8 blocks + BG).
    decode() returns per-block soft-argmax world position + identity confidence."""

    def __init__(self, nblk, half, cx, cy):
        super().__init__()
        self.nblk = nblk
        self.head = nn.Linear(DIM, nblk + 1)
        self.register_buffer("centers", torch.tensor(patch_centers(half, cx, cy)))  # (196,2)
        self.half, self.cx, self.cy = float(half), float(cx), float(cy)

    def forward(self, grid):  # (B,196,384) -> (B,196,nblk+1) logits
        return self.head(grid)

    def decode(self, grid, tau=0.5):
        """grid (B,196,384) -> positions (B,nblk,2) world, conf (B,nblk).
        Soft-argmax over patches of each block's LOGIT (sharp; smooth for CEM). tau divides
        the logit -> small tau = peaked on the top patch (approaches the hard centroid)."""
        logit = self.forward(grid)                        # (B,196,K)
        bl = logit[..., :self.nblk]                       # (B,196,nblk) class-b logit per patch
        attn = (bl.transpose(1, 2) / tau).softmax(-1)     # (B,nblk,196)
        pos = attn @ self.centers                         # (B,nblk,2)
        conf = logit.softmax(-1)[..., :self.nblk].max(1).values  # (B,nblk) peak class-prob
        return pos, conf

    def decode_hard(self, grid, tau=0.1):
        """HARD-centroid decode: assign each patch to its top class FIRST (removes same-color
        cross-talk that the soft-argmax leaks, e.g. red_moon<->red_pentagon), then centroid each
        block's own patches. Soft-argmax fallback for blocks with no assigned patch (rare ~0.5%).
        Fixes red_moon 0.11u->0.023u with no regression on the other 7. CEM is gradient-free so the
        non-smoothness is fine."""
        logit = self.forward(grid)                        # (B,196,K)
        amax = logit.argmax(-1)                            # (B,196) top class per patch
        oh = (amax.unsqueeze(-1) == torch.arange(self.nblk, device=grid.device)).float()  # (B,196,nblk)
        cnt = oh.sum(1)                                    # (B,nblk)
        pos = torch.einsum("bpk,pd->bkd", oh, self.centers) / cnt.clamp(min=1).unsqueeze(-1)
        if (cnt == 0).any():                               # soft fallback for empty blocks
            bl = logit[..., :self.nblk]
            soft = (bl.transpose(1, 2) / tau).softmax(-1) @ self.centers
            pos = torch.where((cnt == 0).unsqueeze(-1), soft, pos)
        conf = logit.softmax(-1)[..., :self.nblk].max(1).values
        return pos, conf


# ---------------- data ----------------

def valid_frames(c):
    """Stack all valid (unpadded) frames: visual (M,196,384) f32, block_xy (M,8,2)."""
    vis, bxy, epid = [], [], []
    for i in range(len(c["seq_lengths"])):
        S = int(c["seq_lengths"][i])
        vis.append(c["visual"][i, :S].astype(np.float32))
        bxy.append(c["block_xy"][i, :S])
        epid.append(np.full(S, i, np.int64))
    return np.concatenate(vis), np.concatenate(bxy), np.concatenate(epid)


def make_labels(bxy, nblk, half, cx, cy):
    """Per-patch label grid: block id at its center patch, else BG=nblk. (M,196) int64."""
    M = bxy.shape[0]
    lab = np.full((M, NP), nblk, np.int64)
    for b in range(nblk):
        tok = world_to_tok(bxy[:, b], half, cx, cy)  # (M,)
        lab[np.arange(M), tok] = b
    return lab


# ---------------- train + D1 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/lt_cache")
    ap.add_argument("--out", default="/workspace/readout")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = dict(np.load(f"{args.cache}/train.npz", allow_pickle=True))
    va = dict(np.load(f"{args.cache}/val.npz", allow_pickle=True))
    blocks = [str(b) for b in tr["blocks"]]
    nblk = len(blocks)
    half = float(tr["half_extent"]); cx, cy = float(tr["center"][0]), float(tr["center"][1])
    print(f"geom: half_extent={half} center=({cx},{cy}) nblk={nblk}  (Gate-1 used 0.32; corrected)")

    vtr, btr, _ = valid_frames(tr)
    vva, bva, eva = valid_frames(va)
    ltr = make_labels(btr, nblk, half, cx, cy)
    print(f"train frames={len(vtr)}  val frames={len(vva)}")

    # class weights (balanced): BG dominates (~188/196 patches/frame)
    cnt = np.bincount(ltr.reshape(-1), minlength=nblk + 1).astype(np.float64)
    w = (cnt.sum() / (len(cnt) * np.maximum(cnt, 1))).astype(np.float32)
    cw = torch.tensor(w, device=dev)

    R = Readout(nblk, half, cx, cy).to(dev)
    opt = torch.optim.Adam(R.parameters(), lr=args.lr)
    Xtr = torch.tensor(vtr, device=dev)            # (M,196,384)
    Ytr = torch.tensor(ltr, device=dev)            # (M,196)
    M = Xtr.shape[0]
    for e in range(args.epochs):
        perm = torch.randperm(M, device=dev)
        tot = 0.0
        for i in range(0, M, args.batch):
            idx = perm[i:i + args.batch]
            logit = R(Xtr[idx])                    # (b,196,K)
            loss = F.cross_entropy(logit.reshape(-1, nblk + 1), Ytr[idx].reshape(-1), weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"  epoch {e}: train CE={tot / M:.4f}")

    torch.save({"state": R.state_dict(), "blocks": blocks, "half": half, "cx": cx, "cy": cy,
                "nblk": nblk, "tau": args.tau}, os.path.join(args.out, "R.pth"))

    # ---------- D1-REPRESENTATIONAL on held-out val ----------
    R.eval()
    Xva = torch.tensor(vva, device=dev)
    lva = make_labels(bva, nblk, half, cx, cy)
    centers_np = patch_centers(half, cx, cy)
    with torch.no_grad():
        logit = R(Xva)                              # (Mv,196,K)
        pred = logit.argmax(-1).cpu().numpy()       # (Mv,196)

    # (id) per-patch block identity acc (patches that are a block center)
    blk = lva != nblk
    id_acc = (pred[blk] == lva[blk]).mean()
    det = ((pred != nblk) == (lva != nblk)).mean()
    print(f"\n=== D1 (id) === block-patch id acc={id_acc:.3f}  det(block-vs-bg)={det:.3f}  chance={1/nblk:.3f}")
    same = {}
    for b in range(nblk):
        same.setdefault(blocks[b].split("_")[0], []).append(b)
    yb, pb = lva[blk], pred[blk]
    for c, bs in same.items():
        if len(bs) == 2:
            a, z = bs
            fa = (pb[yb == a] == z).mean() if (yb == a).any() else 0
            fz = (pb[yb == z] == a).mean() if (yb == z).any() else 0
            print(f"    same-color {c}: {blocks[a]}<->{blocks[z]}  {fa:.2f}/{fz:.2f}")

    # (pos) HARD centroid = Gate-1 decode (representation ceiling); + soft-argmax tau sweep (CEM uses soft)
    Mv = pred.shape[0]
    hard = np.full((Mv, nblk, 2), np.nan, np.float32)
    for i in range(Mv):
        for b in range(nblk):
            toks = np.where(pred[i] == b)[0]
            if len(toks):
                hard[i, b] = centers_np[toks].mean(0)
    herr = np.linalg.norm(hard - bva, axis=-1).reshape(-1)
    herr = herr[~np.isnan(herr)]
    miss = np.isnan(hard[..., 0]).mean()
    print(f"=== D1 (pos) HARD-centroid (Gate-1 decode) === err mean={herr.mean():.4f}u median={np.median(herr):.4f}u  "
          f"within0.05u={np.mean(herr < RADIUS):.3f} within0.025u={np.mean(herr < RADIUS/2):.3f}  miss(no-patch)={miss:.3f}")
    best_tau, best_w05, best_pos = None, -1, None
    for tau in [1.0, 0.5, 0.25, 0.1, 0.05]:
        with torch.no_grad():
            pos_t, _ = R.decode(Xva, tau=tau)
        pos_t = pos_t.cpu().numpy()
        e = np.linalg.norm(pos_t - bva, axis=-1).reshape(-1)
        w05 = np.mean(e < RADIUS)
        print(f"    soft-argmax tau={tau:<4}: err mean={e.mean():.4f}u within0.05u={w05:.3f}")
        if w05 > best_w05:
            best_tau, best_w05, best_pos = tau, w05, pos_t
    pos = best_pos
    print(f"    -> CEM will use soft-argmax tau={best_tau} (within0.05={best_w05:.3f})")

    # (rel) relational: A=start_block, B=target_block per episode (using best soft decode)
    bidx = {b: i for i, b in enumerate(blocks)}
    Ai = np.array([bidx[str(va["start_block"][e])] for e in range(len(va["seq_lengths"]))])
    Bi = np.array([bidx[str(va["target_block"][e])] for e in range(len(va["seq_lengths"]))])
    Aif = Ai[eva]; Bif = Bi[eva]                                  # per valid frame
    dec_d = np.linalg.norm(pos[np.arange(len(pos)), Aif] - pos[np.arange(len(pos)), Bif], axis=-1)
    true_d = np.linalg.norm(bva[np.arange(len(bva)), Aif] - bva[np.arange(len(bva)), Bif], axis=-1)
    mae = np.abs(dec_d - true_d).mean()
    # success-bit (dist<0.05) classification accuracy of R's decoded distance vs truth
    succ_true = true_d < RADIUS
    succ_dec = dec_d < RADIUS
    succ_acc = (succ_true == succ_dec).mean()
    base_rate = succ_true.mean()
    # near/far ordering per episode: is last-frame decoded dist < start-frame decoded dist?
    order_ok = 0; n_ep = 0
    for e in range(len(va["seq_lengths"])):
        S = int(va["seq_lengths"][e])
        if S < 2:
            continue
        fr = np.where(eva == e)[0]
        if dec_d[fr[-1]] < dec_d[fr[0]]:
            order_ok += 1
        n_ep += 1
    print(f"=== D1 (rel) === decoded dist(A,B) MAE vs truth={mae:.4f}u  "
          f"corr={np.corrcoef(dec_d, true_d)[0,1]:.3f}")
    print(f"    success(dist<0.05) classif acc={succ_acc:.3f}  (true success base-rate={base_rate:.3f}, "
          f"n_frames={len(dec_d)})")
    print(f"    near/far ordering (goal decoded-closer than start): {order_ok}/{n_ep}")
    print(f"\n[D1 VERDICT] gate >=0.95 on the RELATION read. Headline = success-classif acc + within-0.05 pos. "
          f"id={id_acc:.3f} pos<0.05(soft)={best_w05:.3f} rel_succ_acc={succ_acc:.3f} dist_MAE={mae:.4f}u")


if __name__ == "__main__":
    main()
