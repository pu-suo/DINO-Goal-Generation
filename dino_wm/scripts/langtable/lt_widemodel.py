"""Capacity check: does a WIDER predictor shrink the COMPOUNDING GAP (free-run - teacher-forced)?

The TF-split proved the residual drift is compounding, not 1-step difficulty (flat TF at ceiling).
But the gap = error-amplification = a property of the map's smoothness/Lipschitz along rollouts,
which is INDEPENDENT of 1-step accuracy -> a bigger model could have identical flat TF yet a smaller
gap. This varies ONLY width (same 3k data, same H=12 warm-start recipe, same frozen DINOv2):
  phase 1: train wider 1-step (teacher-forced) from scratch
  phase 2: warm-start rollout fine-tune (H=12) -- same recipe as the 19M
  phase 3: TF vs free-run sweep -> THE METRIC IS THE GAP curve vs the 19M baseline.
gap shrinks at K8-12 => capacity buys rollout robustness, floor moves (K3->K6 => hierarchy premature).
gap unchanged => amplification is capacity-invariant => floor is real (stronger (b)).

Run: python lt_widemodel.py --cache /workspace/lt_cache_3k --out /workspace/g2_3k_wide \
     --readout /workspace/readout_3k/R.pth --depth 12 --heads 8 --mlp_dim 4096
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/dino_goal/dino_wm")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt_g2 import Dyn, NP, sample_windows  # noqa: E402
from lt_readout import Readout             # noqa: E402

TAU = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/lt_cache_3k")
    ap.add_argument("--out", default="/workspace/g2_3k_wide")
    ap.add_argument("--readout", default="/workspace/readout_3k/R.pth")
    ap.add_argument("--init_base", default="")  # load saved wider 1-step base + skip phase 1 (resume after OOM)
    ap.add_argument("--num_hist", type=int, default=3)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp_dim", type=int, default=4096)
    ap.add_argument("--base_epochs", type=int, default=25)
    ap.add_argument("--base_iters", type=int, default=1000)
    ap.add_argument("--base_batch", type=int, default=20)
    ap.add_argument("--roll_epochs", type=int, default=10)
    ap.add_argument("--roll_iters", type=int, default=600)
    ap.add_argument("--roll_batch", type=int, default=4)
    ap.add_argument("--roll_accum", type=int, default=1)  # grad-accum: effective batch = roll_batch*roll_accum
    ap.add_argument("--roll_H", type=int, default=12)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    tr = dict(np.load(f"{a.cache}/train.npz", allow_pickle=True))
    va = dict(np.load(f"{a.cache}/val.npz", allow_pickle=True))
    blocks = [str(b) for b in tr["blocks"]]; bidx = {b: i for i, b in enumerate(blocks)}
    fs = int(tr["frameskip"]); nh = a.num_hist

    def vstack(c, k):
        return np.concatenate([c[k][i, :int(c["seq_lengths"][i])] for i in range(len(c["seq_lengths"]))], 0).reshape(-1, c[k].shape[-1])
    pm, ps = vstack(tr, "proprio").mean(0), vstack(tr, "proprio").std(0) + 1e-6
    am, as_ = vstack(tr, "actions").mean(0), vstack(tr, "actions").std(0) + 1e-6
    for c in (tr, va):
        c["proprio_n"] = ((c["proprio"] - pm) / ps).astype(np.float32)
        c["actions_n"] = ((c["actions"] - am) / as_).astype(np.float32)
    pm_t = torch.tensor(pm, device=dev); ps_t = torch.tensor(ps, device=dev)
    am_t = torch.tensor(am, device=dev); as_t = torch.tensor(as_, device=dev)
    nparam = lambda m: sum(p.numel() for p in m.parameters())

    def mk():
        return Dyn(nh, 1, fs, depth=a.depth, heads=a.heads, mlp_dim=a.mlp_dim).to(dev)

    def batch(c, n, nf):
        v, p, act = sample_windows(c, nf, n, rng)
        return (torch.tensor(v, device=dev), torch.tensor(p, device=dev), torch.tensor(act, device=dev))

    # ---------- phase 1: wider 1-step (teacher-forced) ----------
    m = mk()
    print(f"WIDE model: depth={a.depth} heads={a.heads} mlp={a.mlp_dim} -> params={nparam(m)/1e6:.1f}M (vs 19M)")
    if a.init_base:
        m.load_state_dict(torch.load(a.init_base, map_location=dev)["model"])
        print(f"  loaded base {a.init_base}; SKIP phase 1")
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
        for e in range(1, a.base_epochs + 1):
            m.train(); tot = 0.0
            for _ in range(a.base_iters):
                loss, _, _ = m.tf_loss(*batch(tr, a.base_batch, nh + 1))
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if e % 5 == 0 or e == a.base_epochs:
                m.eval()
                with torch.no_grad():
                    vl, pv, tv = m.tf_loss(*batch(va, 64, nh + 1))
                print(f"  [1-step] epoch {e}: train MSE={tot/a.base_iters:.4f} val MSE={vl.item():.4f} "
                      f"patchL2={torch.linalg.norm(pv-tv,dim=-1).mean():.3f}")
        torch.save({"model": m.state_dict(), "arch": {"depth": a.depth, "heads": a.heads, "mlp_dim": a.mlp_dim}},
                   os.path.join(a.out, "base.pth"))

    # ---------- phase 2: warm-start rollout fine-tune (H=roll_H) ----------
    H = a.roll_H; opt = torch.optim.AdamW(m.parameters(), lr=2e-4)

    def rollout_loss(vis, prop_n, act_n):
        vlist = [vis[:, j] for j in range(nh)]; losses = []
        for h in range(H):
            t = nh + h
            pred = m.predict(m.assemble(torch.stack(vlist[-nh:], 1), prop_n[:, t-nh:t], act_n[:, t-nh:t]))[:, -1, :NP]
            losses.append(((pred - vis[:, t]) ** 2).mean()); vlist.append(pred)
        return torch.stack(losses).mean()

    for e in range(1, a.roll_epochs + 1):
        m.train(); tot = 0.0
        for _ in range(a.roll_iters):
            opt.zero_grad()
            for _acc in range(a.roll_accum):
                loss = rollout_loss(*batch(tr, a.roll_batch, nh + H)) / a.roll_accum
                loss.backward(); tot += loss.item()
            opt.step()
        print(f"  [rollout H={H} eff-batch={a.roll_batch * a.roll_accum}] epoch {e}: train roll-loss={tot/a.roll_iters:.4f}")
        torch.save({"model": m.state_dict(), "arch": {"depth": a.depth, "heads": a.heads, "mlp_dim": a.mlp_dim}},
                   os.path.join(a.out, f"roll_e{e}.pth"))
        torch.save({"model": m.state_dict(), "arch": {"depth": a.depth, "heads": a.heads, "mlp_dim": a.mlp_dim}},
                   os.path.join(a.out, "model.pth"))

    # ---------- phase 3: TF vs free-run GAP sweep ----------
    ck = torch.load(a.readout, map_location=dev)
    R = Readout(ck["nblk"], ck["half"], ck["cx"], ck["cy"]).to(dev); R.load_state_dict(ck["state"]); R.eval()
    A = np.array([bidx[str(va["start_block"][e])] for e in range(len(va["seq_lengths"]))])
    lo = torch.tensor([0.15, -0.3048], device=dev); hi = torch.tensor([0.6, 0.3048], device=dev)
    m.eval()

    def rollK(e, K):
        vis = [torch.tensor(va["visual"][e, j].astype(np.float32), device=dev) for j in range(nh)]
        prop = [torch.tensor(va["proprio"][e, j], device=dev) for j in range(nh)]
        act = [torch.tensor(va["actions"][e, j].astype(np.float32), device=dev) for j in range(nh - 1)]
        ee = prop[-1].clone(); oa = va["actions"][e, nh-1:nh-1+K].astype(np.float32)
        with torch.no_grad():
            for h in range(K):
                ah = torch.tensor(oa[h], device=dev); act.append(ah)
                wv = torch.stack(vis[-nh:])[None]; wp = ((torch.stack(prop[-nh:])-pm_t)/ps_t)[None]; wa = ((torch.stack(act[-nh:])-am_t)/as_t)[None]
                nxt = m.predict(m.assemble(wv, wp, wa))[0, -1, :NP]
                vis.append(nxt); ee = torch.clamp(ee + ah.reshape(fs, 2).sum(0), lo, hi); prop.append(ee)
            pos, _ = R.decode(vis[-1][None], tau=TAU)
        return pos[0, A[e]].cpu().numpy()

    def tfK(e, K):
        t = nh - 1 + K
        wv = torch.tensor(va["visual"][e, t-nh:t].astype(np.float32), device=dev)[None]
        wp = ((torch.tensor(va["proprio"][e, t-nh:t], device=dev) - pm_t)/ps_t)[None]
        wa = ((torch.tensor(va["actions"][e, t-nh:t].astype(np.float32), device=dev) - am_t)/as_t)[None]
        with torch.no_grad():
            pred = m.predict(m.assemble(wv, wp, wa))[0, -1, :NP]
            pos, _ = R.decode(pred[None], tau=TAU)
        return pos[0, A[e]].cpu().numpy()

    print(f"\n=== WIDE model ({nparam(m)/1e6:.0f}M) TF-split sweep (the GAP is the metric) ===")
    print(f"  {'K':>4} {'FREE-run':>9} {'teacher-F':>9} {'gap':>7}   n")
    for K in [1, 2, 4, 6, 8, 12]:
        fe, te, cnt = [], [], 0
        for e in range(len(va["seq_lengths"])):
            if int(va["seq_lengths"][e]) - nh < K:
                continue
            gt = va["block_xy"][e, nh-1+K, A[e]]
            fe.append(np.linalg.norm(rollK(e, K) - gt)); te.append(np.linalg.norm(tfK(e, K) - gt)); cnt += 1
            if cnt >= a.n:
                break
        print(f"  {K:>4} {np.mean(fe):>9.4f} {np.mean(te):>9.4f} {np.mean(fe)-np.mean(te):>7.4f}   {cnt}")
    print("  vs 19M rollout-FT: gap@K8=0.027, gap@K12=0.043, FREE@K8=0.074. "
          "WIDE gap < 19M gap => capacity buys robustness (floor moves); ~same => floor real.")


if __name__ == "__main__":
    main()
