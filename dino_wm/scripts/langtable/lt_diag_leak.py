"""Phase L / L1 -- LEAK PROBE + decorrelation verification (base/dino_wm env; torch-only).

Question (I12/I14): can the GOAL relation be predicted from z_start ALONE -- no actions, no
language -- held out by trajectory? If yes, a system could exploit start-state structure rather
than ground the command. PASS = probe at/near chance (not materially above the majority-class
baseline).

This is NOT the readout/G1 probe (which predicts CURRENT object state). Here the TARGET is the
GOAL tuple (which pair is named), and the input is the START observation only.

Targets ("goal property"); block2block success ||pos_A-pos_B||<0.05 is SYMMETRIC, so the
unordered pair is the core semantic goal:
  - mover A          (8 classes)     start_block
  - anchor B         (8 classes)     target_block
  - unordered {A,B}  (28 classes)    the pair to bring together  <- primary
Two leak channels (give the leak its best chance -> conservative):
  - POS : GROUND-TRUTH start positions block_xy[:,0] -> 16 coords + 28 pairwise dists (44-d).
          This is the UPPER BOUND on any leak: if the goal is unpredictable from perfect
          positions, it is unpredictable from z_start (a noisy function of positions).
  - Z   : the actual z_start latent visual[:,0] (196,384), mean-pooled to 384.
Held out by trajectory: train on train.npz, eval on val.npz (disjoint trajectories).
3 probe seeds; report mean +/- spread vs uniform chance AND the train-majority-class baseline.

Run (base env): python -u lt_diag_leak.py [--cache /workspace/lt_cache_3k]
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(cache, split):
    c = dict(np.load(f"{cache}/{split}.npz", allow_pickle=True))
    blocks = [str(b) for b in c["blocks"]]
    bidx = {b: i for i, b in enumerate(blocks)}
    z0 = c["visual"][:, 0].astype(np.float32)        # (E,196,384) START latent (pre-push)
    pos0 = c["block_xy"][:, 0].astype(np.float32)    # (E,8,2)     START GT positions
    A = np.array([bidx[str(x)] for x in c["start_block"]], np.int64)
    B = np.array([bidx[str(x)] for x in c["target_block"]], np.int64)
    return z0, pos0, A, B, blocks


def train_probe(Xtr, ytr, Xva, yva, nclass, seed, epochs=400, hid=256, wd=1e-4):
    dev = Xtr.device
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hid), nn.ReLU(),
                        nn.Linear(hid, hid), nn.ReLU(),
                        nn.Linear(hid, nclass)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    ytr_t = torch.tensor(ytr, device=dev)
    yva_t = torch.tensor(yva, device=dev)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(net(Xtr), ytr_t).backward()
        opt.step()
    with torch.no_grad():
        tr_acc = (net(Xtr).argmax(-1).cpu().numpy() == ytr).mean()
        va_acc = (net(Xva).argmax(-1).cpu().numpy() == yva).mean()
    return float(va_acc), float(tr_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/lt_cache_3k")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ztr, ptr, Atr, Btr, blocks = load(a.cache, "train")
    zva, pva, Ava, Bva, _ = load(a.cache, "val")
    nblk = len(blocks)
    pairs = list(itertools.combinations(range(nblk), 2))
    pmap = {p: i for i, p in enumerate(pairs)}
    upair = lambda A, B: np.array([pmap[tuple(sorted((int(x), int(y))))] for x, y in zip(A, B)], np.int64)
    Ptr_u, Pva_u = upair(Atr, Btr), upair(Ava, Bva)

    print(f"[leak] n_train={len(Atr)} n_val={len(Ava)} nblk={nblk} npairs={len(pairs)} dev={dev}")
    # sanity: start-distance of the NAMED pair (the only position-dependence is the >0.06u filter)
    d0_tr = np.linalg.norm(ptr[np.arange(len(ptr)), Atr] - ptr[np.arange(len(ptr)), Btr], axis=-1)
    print(f"[leak] named-pair start dist: mean={d0_tr.mean():.3f}u min={d0_tr.min():.3f}u "
          f"(filter rejects <0.06u). frac<0.07={np.mean(d0_tr < 0.07):.3f}")

    def pos_feats(pos):
        E = len(pos)
        flat = pos.reshape(E, -1)
        d = np.stack([np.linalg.norm(pos[:, i] - pos[:, j], axis=-1) for i, j in pairs], 1)
        return np.concatenate([flat, d], 1).astype(np.float32)

    Fp_tr = torch.tensor(pos_feats(ptr), device=dev)
    Fp_va = torch.tensor(pos_feats(pva), device=dev)
    Fz_tr = torch.tensor(ztr.mean(1), device=dev)
    Fz_va = torch.tensor(zva.mean(1), device=dev)

    targets = {
        "moverA(8)":          (Atr, Ava, nblk),
        "anchorB(8)":         (Btr, Bva, nblk),
        "unordered_pair(28)": (Ptr_u, Pva_u, len(pairs)),
    }
    feats = {"POS(GT,upper-bnd)": (Fp_tr, Fp_va), "Z(latent,meanpool)": (Fz_tr, Fz_va)}

    print(f"\n{'target':>20} | {'channel':>20} | {'val_acc':>14} {'train':>7} "
          f"{'unif':>6} {'major':>6} {'margin':>7}")
    worst_margin = -1.0
    for tn, (ytr, yva, nc) in targets.items():
        unif = 1.0 / nc
        maj = float(np.bincount(ytr, minlength=nc).argmax())
        maj_acc = float((yva == np.bincount(ytr, minlength=nc).argmax()).mean())
        for fn, (Xtr, Xva) in feats.items():
            accs = [train_probe(Xtr, ytr, Xva, yva, nc, s) for s in (0, 1, 2)]
            va = np.array([x[0] for x in accs]); tr = np.array([x[1] for x in accs])
            margin = va.mean() - maj_acc
            worst_margin = max(worst_margin, margin)
            print(f"{tn:>20} | {fn:>20} | {va.mean():.3f}+-{va.std():.3f} {tr.mean():>7.3f} "
                  f"{unif:>6.3f} {maj_acc:>6.3f} {margin:>+7.3f}")

    print(f"\n[L1 VERDICT] worst-case margin over majority-class baseline = {worst_margin:+.3f} "
          f"(across all targets x channels, incl. the GT-position upper bound).")
    print(f"  PASS if the probe is not MATERIALLY above majority (margin small, e.g. <~0.05) -> the goal "
          f"is NOT recoverable from the start; the command is load-bearing / the split is decorrelated.")
    print(f"  Any above-chance is expected to come ONLY from the >0.06u not-already-adjacent filter "
          f"(excludes touching pairs); it does not reveal WHICH pair is named.")


if __name__ == "__main__":
    main()
