"""Relational goal energy: CEM cost = h(R_A, R_B) on the predicted latent.

Pusher-INVARIANT (R reads only the 8 block classes; the dot falls to BG), OBJECT-FACTORED
(scores only the named pair A,B + a don't-disturb term on the rest), SIDE-FREE (Language-Table
block2block has no side relation -- PREPOSITIONS are all proximity synonyms; success is
||pos_A - pos_B|| < 0.05). This REPLACES lt_g2's masked latent-L2 (which was killed by DINOv2
global-attention pusher contamination + side-effect over-constraint). h is closed-form graded:
  cost = ||pos_A - pos_B||  (dense, monotone -> 0 at contact)  + lam * disturb(non-target)
         + off-table penalty on A,B.

PRE-FLIGHT battery (plan PROJECT_DEFINITIVE_PLAN.md §8.1, adapted for a FROZEN readout energy):
  [1] energy-monotonicity (overfit analogue): along the TRUE oracle path, h(R_A,R_B) descends
      ~monotonically toward 0 (success). The energy "bottoms out" on ground-truth-quality latents.
  [2] checkpoint round-trip (adapted): save R+config, reload into a fresh module, identical decode.
  [3] end-to-end tiny: g-parser -> R/h energy -> CEM (few iters) -> relational metric, few eps.
  [4] time one CEM plan at real resolution -> multiply to the n>=30 / n=100 sweep.
  [D2-PREVIEW] (the crux): decode R on the dynamics' PREDICTED latent (oracle rollout) vs on the
      REAL final latent vs ground-truth, split MOVING block (A) vs STATIC blocks -- separates
      "dynamics destroys the moving-block signal" (Risk #2) from "readout distribution-shift".

Run (dino_wm env):
  python lt_relplan.py --cache /workspace/lt_cache --model /workspace/g2/model.pth \
                       --readout /workspace/readout/R.pth
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/langtable_kit")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt_g2 import Dyn, NP          # noqa: E402
from lt_readout import Readout     # noqa: E402

RADIUS = 0.05
TAU = 0.1   # soft-argmax temperature (D1: within-0.05=0.965 at tau=0.1)


def valid_stack(c, key):
    return np.concatenate([c[key][i, :int(c["seq_lengths"][i])] for i in range(len(c["seq_lengths"]))], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/lt_cache")
    ap.add_argument("--model", default="/workspace/g2/model.pth")
    ap.add_argument("--readout", default="/workspace/readout/R.pth")
    ap.add_argument("--num_hist", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.0)   # don't-disturb weight (D4 tunes)
    ap.add_argument("--cem_iters", type=int, default=8)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--elites", type=int, default=16)
    ap.add_argument("--n_ep", type=int, default=8)      # tiny for end-to-end pre-flight
    ap.add_argument("--n_d2", type=int, default=36)     # D2-preview over all val
    ap.add_argument("--d2only", action="store_true",    # fast kill-criterion probe (skip CEM)
                    help="D2 + round-trip only; skip the slow CEM. For mid-training checkpoint probing.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tr = dict(np.load(f"{args.cache}/train.npz", allow_pickle=True))
    va = dict(np.load(f"{args.cache}/val.npz", allow_pickle=True))
    blocks = [str(b) for b in tr["blocks"]]; nblk = len(blocks)
    bidx = {b: i for i, b in enumerate(blocks)}
    fs = int(tr["frameskip"]); nh = args.num_hist
    half = float(tr["half_extent"]); cx, cy = float(tr["center"][0]), float(tr["center"][1])

    pm = valid_stack(tr, "proprio").mean(0); ps = valid_stack(tr, "proprio").std(0) + 1e-6
    am = valid_stack(tr, "actions").mean(0); as_ = valid_stack(tr, "actions").std(0) + 1e-6
    pm_t = torch.tensor(pm, device=dev); ps_t = torch.tensor(ps, device=dev)
    am_t = torch.tensor(am, device=dev); as_t = torch.tensor(as_, device=dev)
    lo = torch.tensor([0.15, -0.3048], device=dev); hi = torch.tensor([0.6, 0.3048], device=dev)
    AB = 0.1

    model = Dyn(nh, 1, fs).to(dev)
    model.load_state_dict(torch.load(args.model, map_location=dev)["model"]); model.eval()
    ck = torch.load(args.readout, map_location=dev)
    R = Readout(ck["nblk"], ck["half"], ck["cx"], ck["cy"]).to(dev)
    R.load_state_dict(ck["state"]); R.eval()

    # g-parser: (start_block,target_block) -> indices. Tuple is cached; trivial under FIXED_8.
    Ai = np.array([bidx[str(va["start_block"][e])] for e in range(len(va["seq_lengths"]))])
    Bi = np.array([bidx[str(va["target_block"][e])] for e in range(len(va["seq_lengths"]))])

    def decode(grid):
        return R.decode(grid, tau=TAU)        # pos (B,nblk,2), conf (B,nblk)

    def rollout(vis_hist, prop_hist, act_prefix, ee0, cem_acts):
        B, H = cem_acts.shape[0], cem_acts.shape[1]
        vis = [vis_hist[k].unsqueeze(0).expand(B, -1, -1) for k in range(nh)]
        prop = [prop_hist[k].unsqueeze(0).expand(B, -1) for k in range(nh)]
        act = [act_prefix[k].unsqueeze(0).expand(B, -1) for k in range(nh - 1)]
        ee = ee0.unsqueeze(0).expand(B, -1).clone()
        for h in range(H):
            a = cem_acts[:, h]; act.append(a)
            wv = torch.stack(vis[-nh:], 1)
            wp = (torch.stack(prop[-nh:], 1) - pm_t) / ps_t
            wa = (torch.stack(act[-nh:], 1) - am_t) / as_t
            nxt = model.predict(model.assemble(wv, wp, wa))[:, -1, :NP]
            vis.append(nxt); ee = torch.clamp(ee + a.reshape(B, fs, 2).sum(1), lo, hi); prop.append(ee)
        return vis[-1], ee

    def rel_cost(grid, ai, bi, start_pos=None, lam=0.0):
        pos, conf = decode(grid)                              # (B,nblk,2)
        d = (pos[:, ai] - pos[:, bi]).norm(dim=-1)            # (B,)
        cost = d.clone()
        if start_pos is not None and lam > 0:
            m = torch.ones(nblk, device=grid.device); m[ai] = 0; m[bi] = 0
            dd = ((pos - start_pos.unsqueeze(0)).norm(dim=-1) * m).sum(-1)
            cost = cost + lam * dd
        for I in (ai, bi):
            oob = ((pos[:, I, 0] < lo[0]) | (pos[:, I, 0] > hi[0]) |
                   (pos[:, I, 1] < lo[1]) | (pos[:, I, 1] > hi[1])).float()
            cost = cost + 0.5 * oob
        return cost, d

    # ============== [1] energy-monotonicity along the TRUE oracle path (overfit analogue) ==============
    print("=== PRE-FLIGHT [1] energy-monotonicity along the TRUE oracle path (R on real frames) ===")
    starts, ends, monos, reach = [], [], [], 0
    for e in range(len(va["seq_lengths"])):
        S = int(va["seq_lengths"][e])
        if S < 2:
            continue
        g = torch.tensor(va["visual"][e, :S].astype(np.float32), device=dev)
        with torch.no_grad():
            pos, _ = decode(g)
        d = (pos[:, Ai[e]] - pos[:, Bi[e]]).norm(dim=-1).cpu().numpy()
        starts.append(d[0]); ends.append(d[-1]); monos.append(np.mean(np.diff(d) < 0))
        reach += int(d[-1] < RADIUS)
    print(f"  n={len(starts)}  start_dist={np.mean(starts):.4f}u  end_dist={np.mean(ends):.4f}u  "
          f"end<0.05(success)={reach}/{len(starts)}  mean monotone-frac={np.mean(monos):.2f}")
    print(f"  -> energy bottoms out at contact on GT-quality latents: {'PASS' if reach >= 0.8*len(starts) else 'CHECK'}")

    # ============== [2] checkpoint round-trip (adapted) ==============
    print("=== PRE-FLIGHT [2] checkpoint round-trip (save R -> reload -> identical decode) ===")
    tmp = "/tmp/R_roundtrip.pth"
    torch.save({"state": R.state_dict(), "nblk": ck["nblk"], "half": ck["half"], "cx": ck["cx"], "cy": ck["cy"]}, tmp)
    R2 = Readout(ck["nblk"], ck["half"], ck["cx"], ck["cy"]).to(dev)
    R2.load_state_dict(torch.load(tmp, map_location=dev)["state"]); R2.eval()
    gtest = torch.tensor(va["visual"][0, :int(va["seq_lengths"][0])].astype(np.float32), device=dev)
    with torch.no_grad():
        p1, _ = R.decode(gtest, tau=TAU); p2, _ = R2.decode(gtest, tau=TAU)
    print(f"  max|decode diff| after reload = {(p1 - p2).abs().max().item():.2e}  "
          f"({'PASS' if (p1 - p2).abs().max().item() < 1e-6 else 'FAIL'})")

    # ============== [D2-PREVIEW] R on PREDICTED latent vs REAL vs GT, moving-vs-static ==============
    print("=== D2-PREVIEW: R on dynamics-PREDICTED latent (oracle rollout) vs REAL vs GT ===")
    er_pred_mov, er_real_mov, er_pred_stat, er_real_stat = [], [], [], []
    dist_gt, dist_pred, dist_real = [], [], []
    n_d2 = min(args.n_d2, len(va["seq_lengths"]))
    for e in range(n_d2):
        S = int(va["seq_lengths"][e]); H = S - nh
        if H < 2:
            continue
        vis_hist = torch.tensor(va["visual"][e, :nh].astype(np.float32), device=dev)
        prop_hist = torch.tensor(va["proprio"][e, :nh], device=dev)
        act_prefix = torch.tensor(va["actions"][e, :nh - 1].astype(np.float32), device=dev)
        ee0 = prop_hist[-1]
        oa = torch.tensor(va["actions"][e, nh - 1:nh - 1 + H].astype(np.float32), device=dev)
        if oa.shape[0] < H:
            oa = torch.cat([oa, torch.zeros(H - oa.shape[0], fs * 2, device=dev)])
        with torch.no_grad():
            pf, _ = rollout(vis_hist, prop_hist, act_prefix, ee0, oa[None])   # predicted final grid
            real_final = torch.tensor(va["visual"][e, S - 1].astype(np.float32), device=dev)[None]
            pos_pred, _ = decode(pf); pos_real, _ = decode(real_final)
        pos_pred = pos_pred[0].cpu().numpy(); pos_real = pos_real[0].cpu().numpy()
        gt = va["block_xy"][e, S - 1]                                          # GT final positions
        a, b = Ai[e], Bi[e]
        er_pred_mov.append(np.linalg.norm(pos_pred[a] - gt[a]))                # moving block A
        er_real_mov.append(np.linalg.norm(pos_real[a] - gt[a]))
        for c in range(nblk):
            if c in (a, b):
                continue
            er_pred_stat.append(np.linalg.norm(pos_pred[c] - gt[c]))          # static blocks
            er_real_stat.append(np.linalg.norm(pos_real[c] - gt[c]))
        dist_gt.append(np.linalg.norm(gt[a] - gt[b]))
        dist_pred.append(np.linalg.norm(pos_pred[a] - pos_pred[b]))
        dist_real.append(np.linalg.norm(pos_real[a] - pos_real[b]))
    f = lambda x: float(np.mean(x))
    print(f"  MOVING block A pos-err:  PRED={f(er_pred_mov):.4f}u  REAL={f(er_real_mov):.4f}u  GT-ref")
    print(f"  STATIC blocks pos-err:   PRED={f(er_pred_stat):.4f}u  REAL={f(er_real_stat):.4f}u")
    dg, dp, dr = np.array(dist_gt), np.array(dist_pred), np.array(dist_real)
    print(f"  decoded dist(A,B): GT={dg.mean():.4f}u  PRED={dp.mean():.4f}u  REAL={dr.mean():.4f}u")
    print(f"  success(dist<0.05) @ final: GT={np.mean(dg<RADIUS):.2f}  PRED={np.mean(dp<RADIUS):.2f}  REAL={np.mean(dr<RADIUS):.2f}")
    print(f"  dist-MAE vs GT:  PRED={np.abs(dp-dg).mean():.4f}u  REAL={np.abs(dr-dg).mean():.4f}u")
    print(f"  -> if PRED moving-err >> REAL moving-err, the DYNAMICS is the bottleneck (Risk #2), not R.")

    if args.d2only:
        print("[d2only] kill-criterion probe done -> read 'MOVING block A pos-err PRED' above "
              f"(PASS target <= {RADIUS}u; REAL ceiling ~0.035u).")
        return

    # ============== [3] end-to-end tiny CEM + [4] timing ==============
    print(f"=== PRE-FLIGHT [3] end-to-end tiny CEM (rel energy, lam={args.lam}) + [4] timing ===")
    rows, cem_succ, orc_succ = [], 0, 0
    n_ep = min(args.n_ep, len(va["seq_lengths"]))
    t_plan = None
    for e in range(n_ep):
        S = int(va["seq_lengths"][e]); H = S - nh
        if H < 2:
            continue
        a, b = Ai[e], Bi[e]
        vis_hist = torch.tensor(va["visual"][e, :nh].astype(np.float32), device=dev)
        prop_hist = torch.tensor(va["proprio"][e, :nh], device=dev)
        act_prefix = torch.tensor(va["actions"][e, :nh - 1].astype(np.float32), device=dev)
        ee0 = prop_hist[-1]
        with torch.no_grad():
            start_pos, _ = decode(vis_hist[-1:])
            start_pos = start_pos[0]
            d_start = (start_pos[a] - start_pos[b]).norm().item()
            oa = torch.tensor(va["actions"][e, nh - 1:nh - 1 + H].astype(np.float32), device=dev)
            if oa.shape[0] < H:
                oa = torch.cat([oa, torch.zeros(H - oa.shape[0], fs * 2, device=dev)])
            of, _ = rollout(vis_hist, prop_hist, act_prefix, ee0, oa[None])
            _, d_orc = rel_cost(of, a, b)
            d_orc = d_orc[0].item()
            mu = torch.zeros(H, fs * 2, device=dev); sig = torch.full((H, fs * 2), 0.06, device=dev)
            t0 = time.time()
            for it in range(args.cem_iters):
                pop = torch.clamp(mu[None] + sig[None] * torch.randn(args.pop, H, fs * 2, device=dev), -AB, AB)
                f_grid, _ = rollout(vis_hist, prop_hist, act_prefix, ee0, pop)
                costs, dists = rel_cost(f_grid, a, b, start_pos=start_pos, lam=args.lam)
                idx = costs.topk(args.elites, largest=False).indices
                mu, sig = pop[idx].mean(0), pop[idx].std(0) + 1e-4
                d_cem = dists[idx[0]].item()
            if t_plan is None:
                t_plan = time.time() - t0
        rows.append((d_start, d_cem, d_orc))
        cem_succ += int(d_cem < RADIUS); orc_succ += int(d_orc < RADIUS)
    rows = np.array(rows)
    print(f"  n={len(rows)}  decoded dist(A,B): start={rows[:,0].mean():.4f}u  CEM_final={rows[:,1].mean():.4f}u  "
          f"oracle-roll={rows[:,2].mean():.4f}u")
    print(f"  success(<0.05): CEM={cem_succ}/{len(rows)}  oracle-roll={orc_succ}/{len(rows)}  "
          f"CEM<start={int((rows[:,1]<rows[:,0]).sum())}/{len(rows)}")
    print(f"  [4] one CEM plan = {t_plan:.2f}s (H,pop,iters={H},{args.pop},{args.cem_iters})  "
          f"-> n=30 ~ {30*t_plan/60:.1f} min, n=100 ~ {100*t_plan/60:.1f} min")
    print("\n[PRE-FLIGHT DONE] wiring sound if [1] bottoms out, [2] PASS, [3] runs end-to-end, [4] affordable. "
          "Read D2-PREVIEW to decide D2 (readout vs dynamics bottleneck).")


if __name__ == "__main__":
    main()
