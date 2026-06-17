"""Lever A.1 diagnostic: is red_moon's readout weakness FIXABLE (capacity/balance) or RESOLUTION-BOUND
(frozen 224 features don't carry red_moon vs red_pentagon)?
  (i)  over-capacity MLP readout vs the linear baseline -> red_moon decode err on REAL held-out frames.
  (ii) binary probe red_moon-vs-red_pentagon on frozen patches at their centers -> separability.
Decision (A.2): red_moon <=0.03u with MLP AND probe separates -> readout-fixable (retrain prod R);
                red_moon >=0.06u AND probe ~chance -> resolution-bound -> FLAG-AND-STOP (no re-encode).
Machinery: single-batch overfit + ckpt round-trip on the trained head first.
"""
import os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt_readout import GRID, NP, DIM, RADIUS, patch_centers, world_to_tok, valid_frames, make_labels  # noqa

dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = dict(np.load("/workspace/lt_cache_3k/train.npz", allow_pickle=True))
va = dict(np.load("/workspace/lt_cache_3k/val.npz", allow_pickle=True))
blocks = [str(b) for b in tr["blocks"]]; nblk = len(blocks); bidx = {b: i for i, b in enumerate(blocks)}
half = float(tr["half_extent"]); cx, cy = float(tr["center"][0]), float(tr["center"][1])
centers_np = patch_centers(half, cx, cy)


class Head(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.head = (nn.Sequential(nn.Linear(DIM, hidden), nn.GELU(), nn.Linear(hidden, nblk + 1))
                     if hidden > 0 else nn.Linear(DIM, nblk + 1))
        self.register_buffer("centers", torch.tensor(centers_np))

    def forward(self, g):
        return self.head(g)

    def decode(self, g, tau=0.1):
        bl = self.head(g)[..., :nblk]
        return (bl.transpose(1, 2) / tau).softmax(-1) @ self.centers


vtr, btr, _ = valid_frames(tr); vva, bva, _ = valid_frames(va)
ltr = make_labels(btr, nblk, half, cx, cy)
Xtr = torch.tensor(vtr, device=dev); Ytr = torch.tensor(ltr, device=dev)
Xva = torch.tensor(vva, device=dev)
cnt = np.bincount(ltr.reshape(-1), minlength=nblk + 1).astype(np.float64)
cw = torch.tensor((cnt.sum() / (len(cnt) * np.maximum(cnt, 1))).astype(np.float32), device=dev)
print(f"train frames={len(vtr)} val frames={len(vva)} nblk={nblk}")


def train_head(hidden, epochs=12, batch=256):
    torch.manual_seed(0)
    R = Head(hidden).to(dev); opt = torch.optim.Adam(R.parameters(), lr=1e-3); M = Xtr.shape[0]
    # machinery: single-batch overfit
    ob = (Xtr[:batch], Ytr[:batch])
    for _ in range(200):
        l = F.cross_entropy(R(ob[0]).reshape(-1, nblk + 1), ob[1].reshape(-1), weight=cw)
        opt.zero_grad(); l.backward(); opt.step()
    overfit = l.item()
    R = Head(hidden).to(dev); opt = torch.optim.Adam(R.parameters(), lr=1e-3)
    for e in range(epochs):
        perm = torch.randperm(M, device=dev)
        for i in range(0, M, batch):
            idx = perm[i:i + batch]
            l = F.cross_entropy(R(Xtr[idx]).reshape(-1, nblk + 1), Ytr[idx].reshape(-1), weight=cw)
            opt.zero_grad(); l.backward(); opt.step()
    # ckpt round-trip
    torch.save({"state": R.state_dict(), "hidden": hidden}, "/tmp/head_rt.pth")
    R2 = Head(hidden).to(dev); R2.load_state_dict(torch.load("/tmp/head_rt.pth")["state"])
    with torch.no_grad():
        d = (R.decode(Xva[:8]) - R2.decode(Xva[:8])).abs().max().item()
    return R, overfit, d


def per_block_err(R, tau=0.1):
    with torch.no_grad():
        pos = R.decode(Xva, tau=tau).cpu().numpy()
    return np.linalg.norm(pos - bva, axis=-1).mean(0)   # (nblk,)


for hidden in [0, 512]:
    R, ofit, rt = train_head(hidden)
    err = per_block_err(R)
    tag = "LINEAR (baseline)" if hidden == 0 else f"MLP hidden={hidden} (over-capacity)"
    print(f"\n[{tag}] overfit-loss={ofit:.4f} ckpt-rt-maxdiff={rt:.1e}")
    order = np.argsort(-err)
    for b in order:
        flag = "  <-- red_moon" if blocks[b] == "red_moon" else ""
        print(f"    {blocks[b]:>16}: {err[b]:.4f}u{flag}")
    print(f"    red_moon={err[bidx['red_moon']]:.4f}u  others-mean={np.delete(err,bidx['red_moon']).mean():.4f}u")

# (ii) binary probe red_moon vs red_pentagon on frozen center-patch latents
rm, rp = bidx["red_moon"], bidx["red_pentagon"]
def center_lat(vis, bxy, blk):
    tok = world_to_tok(bxy[:, blk], half, cx, cy)
    return vis[np.arange(len(vis)), tok]
Xb_tr = np.concatenate([center_lat(vtr, btr, rm), center_lat(vtr, btr, rp)], 0)
yb_tr = np.concatenate([np.zeros(len(vtr)), np.ones(len(vtr))])
Xb_va = np.concatenate([center_lat(vva, bva, rm), center_lat(vva, bva, rp)], 0)
yb_va = np.concatenate([np.zeros(len(vva)), np.ones(len(vva))])
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=2000).fit(Xb_tr, yb_tr)
acc = clf.score(Xb_va, yb_va)
print(f"\n[binary probe] red_moon vs red_pentagon (frozen center-patch latents): held-out acc={acc:.3f} (chance 0.5)")
print("VERDICT: MLP red_moon <=0.03u & probe>>0.5 -> READOUT-FIXABLE (retrain prod R). "
      "MLP red_moon >=0.06u & probe~0.5 -> RESOLUTION-BOUND -> flag-and-stop.")
