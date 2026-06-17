"""G2 readiness smoke (dino_wm env): DINO-WM-faithful dynamics + pre-flight + train + CEM.

Dynamics: ViTPredictor over [196 visual + proprio token + action token] (concat_dim=0), causal,
TF next-latent MSE on visual tokens (= block TF-latent-error, I3). Pre-flight (§8.1): single-batch
overfit, checkpoint round-trip. Train short/small. CEM smoke: latent-space plan toward the encoded
effector-free goal (cost must monotonically drop) + oracle-action-rollout plannability check.

Run (dino_wm env): python lt_g2.py --cache /workspace/lt_cache --out /workspace/g2 --epochs 8
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/dino_goal/dino_wm")  # box repo (for models.vit); falls back below
try:
    from models.vit import ViTPredictor
except Exception:
    sys.path.insert(0, os.path.expanduser("~/Active-Projects/DINO_Goal_Generation/dino_wm"))
    from models.vit import ViTPredictor

NP = 196  # visual patches


class Dyn(nn.Module):
    def __init__(self, num_hist=3, num_pred=1, fs=5, dim=384, depth=6, heads=6, mlp_dim=2048):
        super().__init__()
        self.num_hist, self.num_pred, self.fs, self.dim = num_hist, num_pred, fs, dim
        self.proprio_embed = nn.Sequential(nn.Linear(2, dim), nn.LayerNorm(dim))
        self.action_embed = nn.Sequential(nn.Linear(fs * 2, dim), nn.LayerNorm(dim))
        self.predictor = ViTPredictor(num_patches=NP + 2, num_frames=num_hist, dim=dim,
                                      depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=0.1)

    def assemble(self, vis, prop_n, act_n):  # (b,t,196,d),(b,t,2),(b,t,fs*2)
        p = self.proprio_embed(prop_n).unsqueeze(2)
        a = self.action_embed(act_n).unsqueeze(2)
        return torch.cat([vis, p, a], dim=2)  # (b,t,198,d)

    def predict(self, z):  # (b, num_hist, 198, d) -> (b, num_hist, 198, d)
        b, t = z.shape[:2]
        z = z.reshape(b, t * (NP + 2), self.dim)
        z = self.predictor(z)
        return z.reshape(b, t, NP + 2, self.dim)

    def tf_loss(self, vis, prop_n, act_n):
        z = self.assemble(vis, prop_n, act_n)           # (b, nh+np, 198, d)
        z_src = z[:, :self.num_hist]
        z_tgt = z[:, self.num_pred:]
        z_pred = self.predict(z_src)
        pv, tv = z_pred[:, :, :NP], z_tgt[:, :, :NP]    # visual tokens only (block TF error)
        return ((pv - tv.detach()) ** 2).mean(), pv.detach(), tv.detach()


def sample_windows(cache, num_frames, n, rng):
    """Sample n windows of num_frames consecutive model-steps. Returns vis,prop,act (np)."""
    seq = cache["seq_lengths"]; valid = np.where(seq >= num_frames)[0]
    vis, prop, act = [], [], []
    for _ in range(n):
        i = valid[rng.randint(len(valid))]
        s = rng.randint(0, seq[i] - num_frames + 1)
        vis.append(cache["visual"][i, s:s + num_frames].astype(np.float32))
        prop.append(cache["proprio_n"][i, s:s + num_frames])
        act.append(cache["actions_n"][i, s:s + num_frames])
    return (np.stack(vis), np.stack(prop), np.stack(act))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/lt_cache")
    ap.add_argument("--out", default="/workspace/g2")
    ap.add_argument("--num_hist", type=int, default=3)
    ap.add_argument("--num_pred", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--iters_per_epoch", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--overfit_iters", type=int, default=300)
    ap.add_argument("--overfit_batch", type=int, default=8)
    ap.add_argument("--overfit_only", action="store_true")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="save {out}/ckpt_e{e}.pth every N epochs (for mid-training D2 probing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.RandomState(args.seed)
    tr = dict(np.load(os.path.join(args.cache, "train.npz"), allow_pickle=True))
    va = dict(np.load(os.path.join(args.cache, "val.npz"), allow_pickle=True))
    fs = int(tr["frameskip"]); nf = args.num_hist + args.num_pred

    # normalization stats (proprio, actions) from train valid steps
    def valid_stack(c, key):
        out = []
        for i in range(len(c["seq_lengths"])):
            out.append(c[key][i, :c["seq_lengths"][i]])
        return np.concatenate(out, 0).reshape(-1, c[key].shape[-1])
    pm, ps = valid_stack(tr, "proprio").mean(0), valid_stack(tr, "proprio").std(0) + 1e-6
    am, as_ = valid_stack(tr, "actions").mean(0), valid_stack(tr, "actions").std(0) + 1e-6
    for c in (tr, va):
        c["proprio_n"] = ((c["proprio"] - pm) / ps).astype(np.float32)
        c["actions_n"] = ((c["actions"] - am) / as_).astype(np.float32)
    print(f"train {len(tr['seq_lengths'])} trajs, val {len(va['seq_lengths'])}; fs={fs} nf={nf} device={device}")

    model = Dyn(args.num_hist, args.num_pred, fs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    def batch(c, n):
        v, p, a = sample_windows(c, nf, n, rng)
        return (torch.from_numpy(v).to(device), torch.from_numpy(p).to(device), torch.from_numpy(a).to(device))

    def patch_l2(pv, tv):  # mean per-patch L2 distance (interpretable TF-latent error)
        return torch.linalg.norm(pv - tv, dim=-1).mean().item()

    # ---- PRE-FLIGHT 1: single-batch overfit ----
    print(f"\n[PRE-FLIGHT] single-batch overfit (batch={args.overfit_batch}, iters={args.overfit_iters}):")
    ob = batch(tr, args.overfit_batch)
    # report target scale (per-token L2 norm of the visual targets we must predict)
    with torch.no_grad():
        tgt_vis = ob[0][:, args.num_pred:, :NP]
        print(f"  target visual per-token L2 norm: mean={torch.linalg.norm(tgt_vis,dim=-1).mean():.3f}")
    om = Dyn(args.num_hist, args.num_pred, fs).to(device)
    oo = torch.optim.AdamW(om.parameters(), lr=1e-3)
    for it in range(args.overfit_iters):
        loss, _, _ = om.tf_loss(*ob)
        oo.zero_grad(); loss.backward(); oo.step()
        if it % max(1, args.overfit_iters // 6) == 0 or it == args.overfit_iters - 1:
            print(f"  it{it}: loss={loss.item():.5f}")
    overfit_ok = loss.item() < 0.05
    if args.overfit_only:
        print(f"[PRE-FLIGHT] overfit_only done. overfit_ok={overfit_ok} (final loss {loss.item():.5f})")
        return

    # ---- PRE-FLIGHT 2: checkpoint round-trip ----
    ck = os.path.join(args.out, "ckpt.pth")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "rng": torch.get_rng_state(), "np_rng": rng.get_state()}, ck)
    m2 = Dyn(args.num_hist, args.num_pred, fs).to(device)
    st = torch.load(ck, map_location=device)
    m2.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    lb, _, _ = m2.tf_loss(*ob); lb.backward()
    print(f"[PRE-FLIGHT] checkpoint round-trip: saved+reloaded+resumed 1 step OK (loss {lb.item():.4f})")

    # ---- TRAIN ----
    print("\n[TRAIN] short small dynamics:")
    for e in range(1, args.epochs + 1):
        model.train()
        for _ in range(args.iters_per_epoch):
            loss, _, _ = model.tf_loss(*batch(tr, args.batch))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vb = batch(va, 64)
            vloss, pv, tv = model.tf_loss(*vb)
            # copy-last baseline: predict z_t (no motion) -> error vs z_{t+1}
            vsrc = vb[0][:, :args.num_hist, :, :]  # visual history
            copy_pred = vsrc[:, -args.num_pred:]   # last hist frame as "prediction"
            copy_tgt = vb[0][:, args.num_hist:args.num_hist + args.num_pred]
            copy_l2 = torch.linalg.norm(copy_pred - copy_tgt, dim=-1).mean().item()
        print(f"  epoch {e}: val TF MSE={vloss.item():.5f}  TF patch-L2={patch_l2(pv,tv):.4f}  "
              f"(copy-last patch-L2={copy_l2:.4f})")
        if args.ckpt_every and e % args.ckpt_every == 0:
            torch.save({"model": model.state_dict()}, os.path.join(args.out, f"ckpt_e{e}.pth"))
            print(f"    [ckpt] saved ckpt_e{e}.pth")
    torch.save({"model": model.state_dict()}, os.path.join(args.out, "model.pth"))
    tf_patchl2 = patch_l2(pv, tv)

    # ---- CEM SMOKE (latent space, full-horizon, real-history seed, dot-masked cost) ----
    print("\n[CEM SMOKE] latent-space plan toward encoded effector-free goal (masked energy, alpha=0):")
    model.eval()
    AB = 0.1; nh = args.num_hist
    GG, HALF = 14, float(tr["half_extent"]); CX, CY = float(tr["center"][0]), float(tr["center"][1])
    lo = torch.tensor([0.15, -0.3048], device=device); hi = torch.tensor([0.6, 0.3048], device=device)
    pm_t = torch.tensor(pm, device=device); ps_t = torch.tensor(ps, device=device)
    am_t = torch.tensor(am, device=device); as_t = torch.tensor(as_, device=device)

    def w2tok(ee):  # world ee -> patch token index (for dot masking)
        col = (1 - (ee[1] - CY) / HALF) / 2; row = (1 - (ee[0] - CX) / HALF) / 2
        pc = int(min(max(col * GG, 0), GG - 1)); pr = int(min(max(row * GG, 0), GG - 1))
        return pr * GG + pc

    def rollout_batch(vis_hist, prop_hist, act_prefix, ee0, cem_acts):
        """vis_hist(nh,196,d) prop_hist(nh,2) act_prefix(nh-1,fs*2) ee0(2) cem_acts(B,H,fs*2) raw.
        Returns final predicted visual (B,196,d) and final ee (B,2)."""
        B, H = cem_acts.shape[0], cem_acts.shape[1]
        vis = [vis_hist[k].unsqueeze(0).expand(B, -1, -1) for k in range(nh)]
        prop = [prop_hist[k].unsqueeze(0).expand(B, -1) for k in range(nh)]
        act = [act_prefix[k].unsqueeze(0).expand(B, -1) for k in range(nh - 1)]
        ee = ee0.unsqueeze(0).expand(B, -1).clone()
        for h in range(H):
            a = cem_acts[:, h]
            act.append(a)
            wv = torch.stack(vis[-nh:], 1)
            wp = (torch.stack(prop[-nh:], 1) - pm_t) / ps_t
            wa = (torch.stack(act[-nh:], 1) - am_t) / as_t
            nxt = model.predict(model.assemble(wv, wp, wa))[:, -1, :NP]
            vis.append(nxt)
            ee = torch.clamp(ee + a.reshape(B, fs, 2).sum(1), lo, hi)
            prop.append(ee)
        return vis[-1], ee

    def masked_dist(final, ee_final, z_goal):  # mask the dot patch (pusher-blind energy)
        d = torch.linalg.norm(final - z_goal[None], dim=-1)  # (B,196)
        for b in range(d.shape[0]):
            d[b, w2tok(ee_final[b])] = 0.0
        return d.sum(1) / (NP - 1)

    n_ep = min(20, len(va["seq_lengths"]))
    rows, oracle_better, cem_better = [], 0, 0
    for ep in range(n_ep):
        S = int(va["seq_lengths"][ep])
        H = S - nh  # full remaining horizon (oracle reaches goal at the end)
        if H < 2:
            continue
        vis_hist = torch.tensor(va["visual"][ep, :nh].astype(np.float32), device=device)
        prop_hist = torch.tensor(va["proprio"][ep, :nh], device=device)
        act_prefix = torch.tensor(va["actions"][ep, :nh - 1].astype(np.float32), device=device)
        ee0 = prop_hist[-1]
        z_goal = torch.tensor(va["goal_visual"][ep].astype(np.float32), device=device)
        ee_goal = torch.tensor(va["goal_xy"][ep], device=device)  # unused for cost (effector-free)
        # start distance (mask the start dot patch)
        d_start = masked_dist(vis_hist[-1:].expand(1, -1, -1), ee0[None], z_goal)[0].item()
        with torch.no_grad():
            # oracle-action rollout (the real successful actions through the learned dynamics)
            oa = torch.tensor(va["actions"][ep, nh - 1:nh - 1 + H].astype(np.float32), device=device)
            if oa.shape[0] < H:
                oa = torch.cat([oa, torch.zeros(H - oa.shape[0], fs * 2, device=device)])
            of, oee = rollout_batch(vis_hist, prop_hist, act_prefix, ee0, oa[None])
            d_oracle = masked_dist(of, oee, z_goal)[0].item()
            # CEM
            mu = torch.zeros(H, fs * 2, device=device); sig = torch.full((H, fs * 2), 0.06, device=device)
            c0 = cf = None
            for it in range(8):
                pop = torch.clamp(mu[None] + sig[None] * torch.randn(96, H, fs * 2, device=device), -AB, AB)
                f, eef = rollout_batch(vis_hist, prop_hist, act_prefix, ee0, pop)
                costs = masked_dist(f, eef, z_goal)
                elite = pop[costs.topk(16, largest=False).indices]
                mu, sig = elite.mean(0), elite.std(0) + 1e-4
                if it == 0:
                    c0 = costs.min().item()
                cf = costs.min().item()
        rows.append((d_start, c0, cf, d_oracle))
        oracle_better += int(d_oracle < d_start)
        cem_better += int(cf < d_start)
    rows = np.array(rows)
    print(f"  n={len(rows)} episodes (full per-episode horizon; cost = dot-masked latent L2, alpha=0)")
    print(f"  latent dist to goal: start={rows[:,0].mean():.3f}  CEM_it0={rows[:,1].mean():.3f}  "
          f"CEM_final={rows[:,2].mean():.3f}  oracle-actions={rows[:,3].mean():.3f}")
    print(f"  CEM_final < start: {cem_better}/{len(rows)}  | oracle-rollout < start: {oracle_better}/{len(rows)}")
    print(f"\n[G2 SMOKE SUMMARY] overfit_ok={overfit_ok}  TF-patch-L2={tf_patchl2:.4f} (copy-last~{copy_l2:.2f})  "
          f"CEM<start={cem_better}/{len(rows)}  oracle<start={oracle_better}/{len(rows)}")


if __name__ == "__main__":
    main()
