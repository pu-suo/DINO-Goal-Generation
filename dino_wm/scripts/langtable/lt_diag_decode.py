"""Lever A fix test: red_moon's features ARE separable (probe 0.98) but the SOFT-argmax decode leaks
red_moon's logit onto red_pentagon's same-color patches. Does a peakier / HARD-centroid decode (assign
each patch to its top class first, then centroid red_moon's own patches) fix red_moon without regressing
others or raising the no-detection miss rate? No retrain, no re-encode — a decode change only."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt_readout import Readout, GRID, NP, DIM, patch_centers, valid_frames  # noqa

dev = "cuda" if torch.cuda.is_available() else "cpu"
va = dict(np.load("/workspace/lt_cache_3k/val.npz", allow_pickle=True))
ck = torch.load("/workspace/readout_3k/R.pth", map_location=dev)
blocks = ck["blocks"]; nblk = ck["nblk"]; bidx = {b: i for i, b in enumerate(blocks)}
half, cx, cy = ck["half"], ck["cx"], ck["cy"]
R = Readout(nblk, half, cx, cy).to(dev); R.load_state_dict(ck["state"]); R.eval()
centers = torch.tensor(patch_centers(half, cx, cy), device=dev)        # (196,2)
vva, bva, _ = valid_frames(va); X = torch.tensor(vva, device=dev)
bva_t = bva                                                            # (M,nblk,2) np


def soft_err(tau):
    with torch.no_grad():
        pos, _ = R.decode(X, tau=tau)
    e = np.linalg.norm(pos.cpu().numpy() - bva_t, axis=-1)             # (M,nblk)
    return e.mean(0)


def hard_err():
    with torch.no_grad():
        logit = R(X)                                                  # (M,196,K)
        amax = logit.argmax(-1)                                       # (M,196) top class per patch
    amax = amax.cpu().numpy(); cen = centers.cpu().numpy()
    M = amax.shape[0]; err = np.full((M, nblk), np.nan)
    miss = np.zeros(nblk)
    for b in range(nblk):
        for i in range(M):
            toks = np.where(amax[i] == b)[0]
            if len(toks):
                err[i, b] = np.linalg.norm(cen[toks].mean(0) - bva_t[i, b])
            else:
                miss[b] += 1
    return np.nanmean(err, 0), miss / M


print("per-block decode err (val); rm=red_moon idx", bidx["red_moon"])
for tau in [0.1, 0.05, 0.02]:
    e = soft_err(tau)
    print(f"  soft tau={tau:<5}: red_moon={e[bidx['red_moon']]:.4f}u  others-mean={np.delete(e,bidx['red_moon']).mean():.4f}u")
eh, miss = hard_err()
print(f"  HARD-centroid : red_moon={eh[bidx['red_moon']]:.4f}u  others-mean={np.delete(eh,bidx['red_moon']).mean():.4f}u")
print(f"  HARD per-block err: " + "  ".join(f"{blocks[b].split('_')[0][:1]}{blocks[b].split('_')[1][:2]}={eh[b]:.3f}" for b in range(nblk)))
print(f"  HARD no-detect miss rate: red_moon={miss[bidx['red_moon']]:.3f}  others-max={np.delete(miss,bidx['red_moon']).max():.3f}")
print("=> if HARD red_moon <=~0.03u, low miss, others not regressed -> adopt hard decode (cheap fix).")
