"""Horizon-drift diagnostic: moving-block PRED error vs oracle-rollout length K.

Disambiguates the D2-RED branch:
  - err LOW at small K, RISING with K  -> autoregressive drift compounds -> HIERARCHY/MPC is the
    lever (short-horizon subgoals), NOT more data and NOT a bigger 1-step model.
  - err HIGH even at K=1               -> the 1-step dynamics itself is weak -> data/model capacity.

Rolls the TRUE oracle actions K steps through the frozen dynamics, decodes the named (moving) block
with R, compares to GT block_xy at the matched step. REAL ceiling = R on the real frame.

Run (dino_wm env): python lt_d2_horizon.py --cache /workspace/lt_cache_3k \
    --model /workspace/g2_3k/ckpt_e24.pth --readout /workspace/readout_3k/R.pth --ks 1,2,4,8,16
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/langtable_kit")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt_g2 import Dyn, NP          # noqa: E402
from lt_readout import Readout     # noqa: E402

TAU = 0.1


def vstack(c, k):
    return np.concatenate([c[k][i, :int(c["seq_lengths"][i])] for i in range(len(c["seq_lengths"]))], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/lt_cache_3k")
    ap.add_argument("--model", required=True)
    ap.add_argument("--readout", required=True)
    ap.add_argument("--num_hist", type=int, default=3)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--ks", default="1,2,4,8,16")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = dict(np.load(f"{a.cache}/train.npz", allow_pickle=True))
    va = dict(np.load(f"{a.cache}/val.npz", allow_pickle=True))
    blocks = [str(b) for b in tr["blocks"]]; bidx = {b: i for i, b in enumerate(blocks)}
    fs = int(tr["frameskip"]); nh = a.num_hist
    pm = vstack(tr, "proprio").mean(0); ps = vstack(tr, "proprio").std(0) + 1e-6
    am = vstack(tr, "actions").mean(0); as_ = vstack(tr, "actions").std(0) + 1e-6
    pm_t = torch.tensor(pm, device=dev); ps_t = torch.tensor(ps, device=dev)
    am_t = torch.tensor(am, device=dev); as_t = torch.tensor(as_, device=dev)
    lo = torch.tensor([0.15, -0.3048], device=dev); hi = torch.tensor([0.6, 0.3048], device=dev)

    m = Dyn(nh, 1, fs).to(dev); m.load_state_dict(torch.load(a.model, map_location=dev)["model"]); m.eval()
    ck = torch.load(a.readout, map_location=dev)
    R = Readout(ck["nblk"], ck["half"], ck["cx"], ck["cy"]).to(dev); R.load_state_dict(ck["state"]); R.eval()
    A = np.array([bidx[str(va["start_block"][e])] for e in range(len(va["seq_lengths"]))])

    def rollK(e, K):
        vis = [torch.tensor(va["visual"][e, j].astype(np.float32), device=dev) for j in range(nh)]
        prop = [torch.tensor(va["proprio"][e, j], device=dev) for j in range(nh)]
        act = [torch.tensor(va["actions"][e, j].astype(np.float32), device=dev) for j in range(nh - 1)]
        ee = prop[-1].clone()
        oa = va["actions"][e, nh - 1:nh - 1 + K].astype(np.float32)
        with torch.no_grad():
            for h in range(K):
                ah = torch.tensor(oa[h], device=dev)
                act.append(ah)
                wv = torch.stack(vis[-nh:])[None]
                wp = ((torch.stack(prop[-nh:]) - pm_t) / ps_t)[None]
                wa = ((torch.stack(act[-nh:]) - am_t) / as_t)[None]
                nxt = m.predict(m.assemble(wv, wp, wa))[0, -1, :NP]
                vis.append(nxt); ee = torch.clamp(ee + ah.reshape(fs, 2).sum(0), lo, hi); prop.append(ee)
            pos, _ = R.decode(vis[-1][None], tau=TAU)
        return pos[0, A[e]].cpu().numpy()

    def tfK(e, K):
        # TEACHER-FORCED: predict frame t=nh-1+K from GROUND-TRUTH history [t-nh:t] (single forward).
        # = the 1-step error at trajectory position K. Flat across K => 1-step uniformly good (free-run
        # divergence is pure compounding). Climbs with K => later/post-contact states are intrinsically
        # hard even given perfect history => capacity/resolution/data, NOT compounding.
        t = nh - 1 + K
        wv = torch.tensor(va["visual"][e, t - nh:t].astype(np.float32), device=dev)[None]
        wp = ((torch.tensor(va["proprio"][e, t - nh:t], device=dev) - pm_t) / ps_t)[None]
        wa = ((torch.tensor(va["actions"][e, t - nh:t].astype(np.float32), device=dev) - am_t) / as_t)[None]
        with torch.no_grad():
            pred = m.predict(m.assemble(wv, wp, wa))[0, -1, :NP]
            pos, _ = R.decode(pred[None], tau=TAU)
        return pos[0, A[e]].cpu().numpy()

    print(f"model={a.model}")
    print(f"  {'K':>4} {'FREE-run':>9} {'teacher-F':>9} {'gap':>7}   n")
    for K in [int(x) for x in a.ks.split(",")]:
        fe, te, cnt = [], [], 0
        for e in range(len(va["seq_lengths"])):
            if int(va["seq_lengths"][e]) - nh < K:
                continue
            gt = va["block_xy"][e, nh - 1 + K, A[e]]
            fe.append(np.linalg.norm(rollK(e, K) - gt))
            te.append(np.linalg.norm(tfK(e, K) - gt)); cnt += 1
            if cnt >= a.n:
                break
        f_, t_ = float(np.mean(fe)), float(np.mean(te))
        print(f"  {K:>4} {f_:>9.4f} {t_:>9.4f} {f_ - t_:>7.4f}   {cnt}")
    print("  TF flat + FREE climbs => pure COMPOUNDING (floor real). TF also climbs => intrinsic state "
          "difficulty (capacity/resolution/data), NOT compounding -> hierarchy would be premature.")
    real = []
    for e in range(min(a.n, len(va["seq_lengths"]))):
        if int(va["seq_lengths"][e]) < nh + 1:
            continue
        g = torch.tensor(va["visual"][e, int(va["seq_lengths"][e]) - 1].astype(np.float32), device=dev)[None]
        with torch.no_grad():
            pos, _ = R.decode(g, tau=TAU)
        real.append(np.linalg.norm(pos[0, A[e]].cpu().numpy() - va["block_xy"][e, int(va["seq_lengths"][e]) - 1, A[e]]))
    print(f"  REAL ceiling (R on real final frame): {np.mean(real):.4f}u")
    print("  VERDICT: rises with K => autoregressive drift => HIERARCHY/MPC lever; flat-high => 1-step/data lever.")


if __name__ == "__main__":
    main()
