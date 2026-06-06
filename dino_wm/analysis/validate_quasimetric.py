"""Validation gates for the trained quasimetric head -- PASS BEFORE TOUCHING CEM.

On HELD-OUT trajectories (val cache; the head trains on the train cache) confirm:
  (a) MONOTONICITY: d(z_t, z_goal) decreases ~monotonically as t -> goal along a
      trajectory (the VIP-style smoothness check). Goal = last model-step.
  (b) ASYMMETRY:    d(a,b) != d(b,a) meaningfully (scatter cheap, re-assembly dear).
                    If ~symmetric, the head isn't learning irreversibility -> debug.
  (c) SCALE SANITY: d(z_t, z_{t+1}) ~ 1 on adjacent model-steps (the local cost).

Masking is the SAME union-of-pushers helper used in training and planning. Numbers
+ plots are written to <out>; do not proceed to CEM if (a) or (b) fails.

    python analysis/validate_quasimetric.py --qm_ckpt $CKPTS/qm/iqe_d0/qm_head.pth \
        --cache_dir $DATASET_DIR/pusht_noise/qm_latents --split val --out qm_outputs/validate
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.qm_latent_dset import trajectory_views
from env.pusht.multicolor_common import manipulator_energy_mask
from models.quasimetric import load_quasimetric_head


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    ra = (ra - ra.mean()); rb = (rb - rb.mean())
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12
    return float((ra * rb).sum() / denom)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qm_ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="qm_outputs/validate")
    ap.add_argument("--n_traj", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    head, ckpt = load_quasimetric_head(args.qm_ckpt, device=device)
    dil = int(ckpt.get("mask_dilation", 0))
    latents, states, trajs, pxy, meta = trajectory_views(args.cache_dir, args.split)
    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(trajs), size=min(args.n_traj, len(trajs)), replace=False)

    def keep(gi, gj):
        return torch.from_numpy(manipulator_energy_mask([pxy[gi], pxy[gj]], dilation=dil)).to(device)

    def d(gi, gj):
        za = latents[gi].unsqueeze(0).to(device); zb = latents[gj].unsqueeze(0).to(device)
        return float(head(za, zb, keep(gi, gj))[0])

    # (a) monotonicity to the (per-traj) goal = last model-step
    mono_fracs, spearmans, curves = [], [], []
    for tid in pick:
        s, L = trajs[tid]
        g = s + L - 1
        ds = [d(s + t, g) for t in range(L)]
        steps_to_go = [L - 1 - t for t in range(L)]
        diffs = np.diff(ds)
        mono_fracs.append(float(np.mean(diffs < 0)))   # decreasing as t->goal
        spearmans.append(spearman(steps_to_go, ds))    # +1 = perfect (more steps => larger d)
        curves.append(ds)
    mono = dict(decreasing_frac_mean=float(np.mean(mono_fracs)),
                decreasing_frac_min=float(np.min(mono_fracs)),
                spearman_steps_vs_d_mean=float(np.mean(spearmans)))

    # (b) asymmetry: forward (start->late) vs backward (late->start)
    fwd, bwd = [], []
    for tid in pick:
        s, L = trajs[tid]
        for off in (L - 1, max(1, L // 2)):
            fwd.append(d(s, s + off)); bwd.append(d(s + off, s))
    fwd = np.asarray(fwd); bwd = np.asarray(bwd)
    asym = dict(mean_fwd=float(fwd.mean()), mean_bwd=float(bwd.mean()),
                mean_abs_gap=float(np.mean(np.abs(fwd - bwd))),
                rel_gap=float(np.mean(np.abs(fwd - bwd)) / (np.mean(np.abs(fwd)) + 1e-9)),
                frac_fwd_lt_bwd=float(np.mean(fwd < bwd)))

    # (c) scale: adjacent model-step distance ~ 1
    adj = []
    for tid in pick:
        s, L = trajs[tid]
        for t in range(L - 1):
            adj.append(d(s + t, s + t + 1))
    adj = np.asarray(adj)
    scale = dict(adjacent_mean=float(adj.mean()), adjacent_std=float(adj.std()),
                 adjacent_p10=float(np.percentile(adj, 10)),
                 adjacent_p90=float(np.percentile(adj, 90)))

    report = dict(qm_ckpt=args.qm_ckpt, split=args.split, n_traj=int(len(pick)),
                  mask_dilation=dil, monotonicity=mono, asymmetry=asym, scale=scale)
    print(json.dumps(report, indent=2))
    with open(out / "validation.json", "w") as f:
        json.dump(report, f, indent=2)

    # gate verdicts (advisory thresholds; report regardless)
    print("\n=== GATES ===")
    print(f"(a) monotonicity: decreasing-frac mean={mono['decreasing_frac_mean']:.2f} "
          f"(min {mono['decreasing_frac_min']:.2f}), spearman={mono['spearman_steps_vs_d_mean']:.2f} "
          f"-> {'PASS' if mono['decreasing_frac_mean'] > 0.7 else 'CHECK'}")
    print(f"(b) asymmetry: rel_gap={asym['rel_gap']:.2f}, fwd={asym['mean_fwd']:.2f} "
          f"bwd={asym['mean_bwd']:.2f} -> {'PASS' if asym['rel_gap'] > 0.05 else 'FAIL (≈symmetric)'}")
    print(f"(c) scale: adjacent d={scale['adjacent_mean']:.2f}±{scale['adjacent_std']:.2f} "
          f"(target ~1) -> {'PASS' if 0.5 < scale['adjacent_mean'] < 2.0 else 'CHECK'}")

    # plots (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for c in curves[:20]:
            ax[0].plot(range(len(c)), c, alpha=0.5)
        ax[0].set_title("(a) d(z_t, z_goal) vs t  (should decrease)")
        ax[0].set_xlabel("model-step t"); ax[0].set_ylabel("d")
        lim = max(fwd.max(), bwd.max())
        ax[1].scatter(bwd, fwd, s=12, alpha=0.6); ax[1].plot([0, lim], [0, lim], "r--")
        ax[1].set_title("(b) forward vs backward d"); ax[1].set_xlabel("d(late->start)")
        ax[1].set_ylabel("d(start->late)")
        ax[2].hist(adj, bins=30); ax[2].axvline(1.0, c="r", ls="--")
        ax[2].set_title("(c) adjacent-step d (target ~1)")
        fig.tight_layout(); fig.savefig(out / "validation.png", dpi=110)
        print(f"\nsaved {out/'validation.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
