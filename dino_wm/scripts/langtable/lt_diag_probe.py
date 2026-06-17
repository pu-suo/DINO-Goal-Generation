"""Lever A.1(ii) torch-only: do the frozen 224 features separate red_moon from red_pentagon at their
center patches? Also detection coverage: how often does the linear readout even FIRE red_moon at its
patch. Decides resolution-bound (probe ~chance) vs a coverage/detection issue (probe high)."""
import os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt_readout import DIM, world_to_tok, valid_frames  # noqa

dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = dict(np.load("/workspace/lt_cache_3k/train.npz", allow_pickle=True))
va = dict(np.load("/workspace/lt_cache_3k/val.npz", allow_pickle=True))
blocks = [str(b) for b in tr["blocks"]]; bidx = {b: i for i, b in enumerate(blocks)}
half = float(tr["half_extent"]); cx, cy = float(tr["center"][0]), float(tr["center"][1])
vtr, btr, _ = valid_frames(tr); vva, bva, _ = valid_frames(va)


def center_lat(vis, bxy, blk):
    tok = world_to_tok(bxy[:, blk], half, cx, cy)
    return vis[np.arange(len(vis)), tok]


def probe(a_name, b_name):
    a, b = bidx[a_name], bidx[b_name]
    Xtr = np.concatenate([center_lat(vtr, btr, a), center_lat(vtr, btr, b)], 0)
    ytr = np.concatenate([np.zeros(len(vtr)), np.ones(len(vtr))])
    Xva = np.concatenate([center_lat(vva, bva, a), center_lat(vva, bva, b)], 0)
    yva = np.concatenate([np.zeros(len(vva)), np.ones(len(vva))])
    Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, device=dev, dtype=torch.float32)
    clf = nn.Linear(DIM, 1).to(dev); opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for _ in range(300):
        l = F.binary_cross_entropy_with_logits(clf(Xtr_t).squeeze(-1), ytr_t)
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        pred = (clf(torch.tensor(Xva, device=dev)).squeeze(-1) > 0).cpu().numpy()
    return (pred == yva).mean()


print("Binary separability (frozen center-patch latents, held-out val acc; chance=0.5):")
for pair in [("red_moon", "red_pentagon"), ("blue_moon", "blue_cube"),
             ("green_cube", "green_star"), ("yellow_star", "yellow_pentagon")]:
    print(f"  {pair[0]:>16} vs {pair[1]:<16}: {probe(*pair):.3f}")
print("=> red_moon/red_pentagon ~0.5 => features don't separate => RESOLUTION-BOUND (confirms the MLP")
print("   result). >>0.5 => features separate but the multi-class readout under-detects red_moon.")
