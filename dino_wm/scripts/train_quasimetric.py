"""Offline QRL training of the pure-V* quasimetric cost-to-go head.

Borrows the QRL dual (min-max) objective and softplus-phi reweighting from
quasimetric-rl (Wang et al. 2023), adapted to VALUE-ONLY: no latent transition
model T, no Q, no policy -- we plan with the frozen DINO-WM dynamics + CEM, so we
only need d_theta(z, z_goal).

Objective (constant local cost r = -1 per model step):

    min_theta  max_{lambda>=0}
        -E_{s,g}[ phi(d_theta(z_s, z_g)) ]
        + lambda * ( E_{(z,z')}[ relu(d_theta(z, z') - 1)^2 ] - eps^2 )

    phi(x) = -softplus(OFFSET - x, beta)   (monotone-increasing, saturating)

lambda is softplus-parameterized to stay >= 0 and ascended with its own optimizer
(dual ascent). After training, -d_theta approximates V* in model-step units.

Verified hyperparameters (research report / QRL paper) are the argparse defaults:
eps=0.25 (NOT horizon-scaled), lambda init 0.01 / lr 0.01, model lr 1e-4, batch 256,
IQE-maxmean 64x32 (proj_out 2048). Set OFFSET near the cache's p90/max model-step
trajectory length (printed by cache_qm_latents.py and echoed below).

Run (GPU box):
    cd dino_wm && source $WS/activate.sh
    python scripts/train_quasimetric.py --out $CKPTS/qm/iqe_d0 --steps 60000
Mac smoke (cpu, tiny):
    .../dino_wm_dev/bin/python scripts/train_quasimetric.py --smoke
"""
import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.qm_latent_dset import QMLatentDataset
from models.quasimetric import build_quasimetric_head


def inv_softplus(y):
    # x such that softplus(x) = y  (y > 0)
    return math.log(math.expm1(y))


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.RandomState(1234 + worker_id)


def cycle(loader):
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir",
                    default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_noise", "qm_latents"))
    ap.add_argument("--out", default="qm_outputs/iqe_d0")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num_workers", type=int, default=0)
    # head
    ap.add_argument("--head_type", default="iqe", choices=["iqe", "mrn", "mrn_fixed", "sym_l2"])
    ap.add_argument("--proj_out", type=int, default=2048)
    ap.add_argument("--dim_per_component", type=int, default=32)
    ap.add_argument("--f_out", type=int, default=256)
    ap.add_argument("--proj_hidden", type=int, default=512)
    ap.add_argument("--no_mask_channel", action="store_true")
    # QRL objective
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--lambda_init", type=float, default=0.01)
    ap.add_argument("--lambda_lr", type=float, default=0.01)
    ap.add_argument("--model_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="AdamW weight decay on the head; a small value (e.g. 1e-4) curbs the "
                         "overfitting that sank iqe_d0 (val adjacent-d ~9x train). 0 == plain Adam.")
    ap.add_argument("--phi_offset", type=float, default=30.0,
                    help="OFFSET in phi=-softplus(OFFSET-x,beta); set near cache p90/max model-steps.")
    ap.add_argument("--phi_beta", type=float, default=0.1)
    ap.add_argument("--drop_relu", action="store_true",
                    help="QRL App.A constant-cost form: use (d-1)^2 directly (no relu).")
    # sampling / mask
    ap.add_argument("--mask_dilation", type=int, default=0)
    ap.add_argument("--p_random_goal", type=float, default=0.0)
    ap.add_argument("--max_goal_offset", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--val_every", type=int, default=1000,
                    help="every N steps, report ABSOLUTE held-out health on the VAL cache "
                         "(gate-c scale band, d_sg-collapse, and a gate-(a) MONOTONICITY proxy) "
                         "and save qm_head_bestval.pth at the BEST held-out monotonicity. iqe_d0 "
                         "overfit silently; this surfaces the decisive ordering signal live. 0 disables.")
    ap.add_argument("--val_mono_k", type=int, default=8,
                    help="#val trajectories for the live monotonicity proxy (cheap; the full "
                         "analysis/validate_quasimetric.py uses 40 and is the real arbiter).")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU smoke run on the smoke cache")
    args = ap.parse_args()

    if args.smoke:
        args.device = "cpu"; args.steps = 30; args.batch = 8; args.log_every = 5
        args.cache_dir = args.cache_dir.replace("pusht_noise", "pusht_noise_smoke")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out}")

    dset = QMLatentDataset(args.cache_dir, "train", mask_dilation=args.mask_dilation,
                           p_random_goal=args.p_random_goal, max_goal_offset=args.max_goal_offset)
    print(f"[cache meta] model-step traj len: p50={dset.meta.get('model_step_len_p50')} "
          f"p90={dset.meta.get('model_step_len_p90')} max={dset.meta.get('model_step_len_max')}  "
          f"--> phi_offset={args.phi_offset}")
    loader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn if args.num_workers else None)
    it = cycle(loader)

    # LIVE val monitor. iqe_d0 overfit silently and we only caught it at the post-hoc gate
    # after hours. Each --val_every steps we report ABSOLUTE, gate-aligned held-out health:
    #   - val adjacent-d in the gate-(c) band 0.5..2.0   (scale)
    #   - val d_sg not collapsed (>3)                     (spreading didn't die)
    #   - val MONOTONICITY proxy (gate-(a)): the decisive signal -- d(z_t, z_goal) should
    #     DECREASE toward the goal. The level metrics (d_trans/d_sg) are BLIND to this
    #     (d_sg is phi-saturated and uses random cross-traj anchors), and a val/train RATIO
    #     is misleading (it inflates when TRAIN d_trans collapses, which the one-sided
    #     constraint permits, even when val is fine). bestval is chosen by MONOTONICITY,
    #     not scale -- that is what gate (a) and CEM ordering actually need.
    # Best-effort: disabled cleanly if the val cache is absent/tiny.
    val_it = None
    val_dset = None
    if args.val_every > 0 and not args.smoke:
        try:
            vds_tmp = QMLatentDataset(args.cache_dir, "val", mask_dilation=args.mask_dilation,
                                      p_random_goal=args.p_random_goal, max_goal_offset=args.max_goal_offset)
            if len(vds_tmp) > 0:
                val_dset = vds_tmp
                vb = min(args.batch, len(val_dset))
                val_loader = torch.utils.data.DataLoader(val_dset, batch_size=vb, shuffle=True,
                                                         drop_last=False, num_workers=0)
                val_it = cycle(val_loader)
            else:
                print("[val] val cache has 0 transitions; monitoring disabled")
        except Exception as e:
            print(f"[val] monitoring disabled (no/unreadable val cache: {e})")
    best_val_mono = -1.0   # maximize held-out monotonicity (gate-(a) proxy)

    head_cfg = dict(head_type=args.head_type, proj_out=args.proj_out,
                    dim_per_component=args.dim_per_component, f_out=args.f_out,
                    proj_hidden=args.proj_hidden, append_mask_channel=not args.no_mask_channel)
    head = build_quasimetric_head(head_cfg).to(device).train()
    n_params = sum(p.numel() for p in head.parameters())
    print(f"head={args.head_type} params={n_params/1e6:.2f}M wd={args.weight_decay}")

    def save_ckpt(path, step, **extra):
        torch.save(dict(state_dict=head.state_dict(), head_cfg=head_cfg, args=vars(args),
                        mask_dilation=args.mask_dilation, phi_offset=args.phi_offset,
                        phi_beta=args.phi_beta, eps=args.eps, step=step, **extra), path)

    @torch.no_grad()
    def val_monotonicity(k):
        """Cheap held-out gate-(a) proxy: mean decreasing-fraction of d(z_{s+t}, z_goal)
        along k val trajectories (goal = last model-step), per-traj batched. Mirrors
        analysis/validate_quasimetric.py gate (a) -- the ORDERING CEM relies on, and the
        one thing the d_trans/d_sg level metrics cannot see. Returns NaN if unavailable."""
        if val_dset is None:
            return float("nan")
        trajs = list(zip(val_dset.starts, val_dset.lengths))
        fr = []
        for s, L in trajs[:min(k, len(trajs))]:
            if L < 3:
                continue
            g = s + L - 1
            za = val_dset.latents[s:s + L].float().to(device)                       # (L,196,384)
            zb = val_dset.latents[g].float().unsqueeze(0).expand(L, -1, -1).to(device)
            keep = torch.stack([val_dset._keep(s + t, g) for t in range(L)]).to(device)  # (L,196)
            ds = head(za, zb, keep).cpu().numpy()                                    # (L,) d(z_{s+t}, z_g)
            fr.append(float((np.diff(ds) < 0).mean()))
        return float(np.mean(fr)) if fr else float("nan")

    lam_raw = torch.nn.Parameter(torch.tensor(float(inv_softplus(args.lambda_init)), device=device))
    opt = torch.optim.AdamW(head.parameters(), lr=args.model_lr, weight_decay=args.weight_decay)
    opt_lam = torch.optim.Adam([lam_raw], lr=args.lambda_lr)
    eps2 = args.eps ** 2

    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = next(it)
        z_a, z_b, keep_ab = b["z_a"].to(device), b["z_b"].to(device), b["keep_ab"].to(device)
        z_s, z_g, keep_sg = b["z_s"].to(device), b["z_g"].to(device), b["keep_sg"].to(device)

        d_trans = head(z_a, z_b, keep_ab)                     # (B,)  ~ local cost (->1)
        d_sg = head(z_s, z_g, keep_sg)                        # (B,)  cost-to-go to maximize

        phi = -F.softplus(args.phi_offset - d_sg, beta=args.phi_beta)
        obj = phi.mean()                                      # maximize
        gap = (d_trans - 1.0)
        viol = gap.pow(2) if args.drop_relu else F.relu(gap).pow(2)
        constraint = viol.mean() - eps2                       # want <= 0
        lam = F.softplus(lam_raw)

        loss_theta = -obj + lam.detach() * constraint
        loss_lam = -(lam * constraint.detach())               # dual ascent on lambda
        loss = loss_theta + loss_lam

        opt.zero_grad(); opt_lam.zero_grad()
        loss.backward()
        opt.step(); opt_lam.step()

        if step % args.log_every == 0 or step == 1:
            rec = dict(step=step, lam=float(lam), d_trans=float(d_trans.mean()),
                       viol=float(viol.mean()), constraint=float(constraint),
                       d_sg=float(d_sg.mean()), obj=float(obj),
                       sps=step / (time.time() - t0))
            hist.append(rec)
            print(f"step {step:6d} | lam {rec['lam']:.4f} | d_trans {rec['d_trans']:.3f} "
                  f"(viol {rec['viol']:.4f}, eps^2 {eps2:.4f}) | d_sg {rec['d_sg']:.3f} | "
                  f"{rec['sps']:.1f} it/s")
            # stability guards (report, don't crash)
            if not math.isfinite(rec["lam"]) or rec["lam"] > 1e4:
                print("  [warn] lambda diverging -> lower --lambda_lr toward 0.003")
            if rec["d_trans"] < 1e-3 or rec["d_sg"] < 1e-3:
                print("  [warn] distances collapsing toward 0 -> check phi_offset / lambda_lr")

        if val_it is not None and (step % args.val_every == 0 or step == 1):
            head.eval()
            with torch.no_grad():
                vbm = next(val_it)
                vdt = head(vbm["z_a"].to(device), vbm["z_b"].to(device), vbm["keep_ab"].to(device))
                vds = head(vbm["z_s"].to(device), vbm["z_g"].to(device), vbm["keep_sg"].to(device))
                v_dt, v_ds = float(vdt.mean()), float(vds.mean())
            v_mono = val_monotonicity(args.val_mono_k)   # held-out gate-(a) proxy (decisive)
            head.train()
            # ABSOLUTE, gate-aligned flags (each maps 1:1 onto a real gate). The old
            # val/train RATIO was misleading -- it inflates when TRAIN d_trans collapses
            # (which the one-sided constraint permits) even while val is healthy, and it is
            # blind to monotonicity, the only thing that decides planning.
            flags = []
            if not (0.5 < v_dt < 2.0):                          # gate (c): scale band
                flags.append("SCALE")
            if v_ds < 3.0:                                      # spreading collapsed
                flags.append("DSG_COLLAPSE")
            if not math.isnan(v_mono) and v_mono < 0.7:         # gate (a): monotonicity
                flags.append("MONO")
            tag = ("  <-- WARN: " + ",".join(flags)) if flags else ""
            mono_s = "nan" if math.isnan(v_mono) else f"{v_mono:.2f}"
            print(f"   [val] d_trans {v_dt:.3f} (gate-c 0.5-2.0) d_sg {v_ds:.3f} "
                  f"mono {mono_s} (gate-a>0.7){tag}")
            # Best = HIGHEST held-out monotonicity (what gate (a) and CEM ordering need),
            # among scale-sane, spread checkpoints. The previous abs(val_d_trans-1) selector
            # was SCALE-only and blind to shape -- it could pin "best" on an
            # under-constrained-drift step that fails monotonicity.
            if (0.5 < v_dt < 2.0) and v_ds > 3.0 and not math.isnan(v_mono) \
                    and v_mono > best_val_mono:
                best_val_mono = v_mono
                save_ckpt(out / "qm_head_bestval.pth", step,
                          val_mono=v_mono, val_d_trans=v_dt, val_d_sg=v_ds)

        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(out / "qm_head.pth", step)        # latest (crash recovery)
            with open(out / "train_log.json", "w") as f:
                json.dump(hist, f, indent=2)

    # final curve plot (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = [h["step"] for h in hist]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(s, [h["lam"] for h in hist]); ax[0].set_title("lambda"); ax[0].set_xlabel("step")
        ax[1].plot(s, [h["d_trans"] for h in hist], label="d_trans (->1)")
        ax[1].plot(s, [h["d_sg"] for h in hist], label="d_sg"); ax[1].legend(); ax[1].set_title("distances")
        ax[2].plot(s, [h["viol"] for h in hist]); ax[2].axhline(eps2, ls="--", c="r")
        ax[2].set_title("constraint violation vs eps^2")
        fig.tight_layout(); fig.savefig(out / "train_curves.png", dpi=110)
        print(f"saved {out/'train_curves.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"Done. final head -> {out/'qm_head.pth'}")
    if (out / "qm_head_bestval.pth").exists():
        print(f"BEST-VAL head -> {out/'qm_head_bestval.pth'} (held-out monotonicity {best_val_mono:.2f}). "
              f"Validate and PLAN with this one -- it has the best held-out cost-to-go ORDERING "
              f"(gate (a)), which is what CEM needs; the final head may have drifted. Still run the "
              f"full analysis/validate_quasimetric.py -- the live proxy uses only {args.val_mono_k} trajs.")


if __name__ == "__main__":
    main()
