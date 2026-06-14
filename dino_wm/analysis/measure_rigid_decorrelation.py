"""Part-1 guard 1.8: measure start<->goal decorrelation for rigid-transform goals.

A rigid transform links start<->goal by the PRESERVED displacement, so some linear
correlation is intrinsic. The question that actually matters for "language is
load-bearing" is whether the GOAL REGION is predictable from the start. We report
BOTH: (a) linear/circular correlation of absolute poses, and (b) the operative
region metrics -- P(goal_cell != start_cell) and the conditional entropy
H(goal_cell | start_cell) vs the marginal H(goal_cell). A text-ignoring model can
only use the start; if H(goal|start) ~ H(goal) and most goals leave the start
cell, language must carry the region.

  python analysis/measure_rigid_decorrelation.py --data_path <pusht_noise> --split train --n_max 6000
"""
import os, sys, argparse, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.rigid_goal_common import (
    traj_uses_wall, sample_valid_transform, make_language, REGION_BOUNDS,
)
from metrics.regional_success import block_cell, block_centroid, REGION_NCELLS


def circ_corr(a, b):
    """Circular correlation coefficient (Jammalamadaka-Sarma) for angles a,b (rad)."""
    a = np.asarray(a); b = np.asarray(b)
    abar = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    bbar = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    sa, sb = np.sin(a - abar), np.sin(b - bbar)
    return float((sa * sb).sum() / np.sqrt((sa**2).sum() * (sb**2).sum()))


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_max", type=int, default=6000, help="cap kept trajectories")
    ap.add_argument("--wall_margin", type=float, default=10.0)
    ap.add_argument("--min_seq", type=int, default=20, help="skip ultra-short trajs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = os.path.join(args.data_path, args.split)
    states = torch.load(os.path.join(d, "states.pth")).double().numpy()
    seq = pickle.load(open(os.path.join(d, "seq_lengths.pkl"), "rb"))
    rng = np.random.RandomState(args.seed)
    print(f"loaded {len(seq)} {args.split} trajs; states {states.shape}")

    n_wall = n_short = n_nofit = 0
    recs = []
    for i in range(len(seq)):
        L = int(seq[i])
        if L < args.min_seq:
            n_short += 1; continue
        if traj_uses_wall(states[i], L, args.wall_margin):
            n_wall += 1; continue
        res = sample_valid_transform(states[i], L, rng, goal_bounds=REGION_BOUNDS)
        if res is None:
            n_nofit += 1; continue
        theta, t, tf = res
        s0, sT = tf[0], tf[L - 1]
        lang = make_language(s0, sT)
        recs.append((s0[2], s0[3], s0[4], sT[2], sT[3], sT[4],
                     block_cell(block_centroid(s0[2:4], s0[4])),  # start cell on centroid too
                     lang["region_cell"], lang["rel_rot_deg"]))
        if len(recs) >= args.n_max:
            break

    n = len(recs)
    sx, sy, sth, gx, gy, gth = (np.array([r[k] for r in recs]) for k in range(6))
    start_cell = [r[6] for r in recs]; goal_cell = [r[7] for r in recs]
    relrot = np.array([r[8] for r in recs])

    print(f"\nkept {n} goals  (excluded: wall={n_wall}, short={n_short}, no-valid-tf={n_nofit})")
    print("\n--- (a) absolute-pose correlation (intrinsic to rigid linking; not the gate) ---")
    print(f"  corr(start_x, goal_x) = {np.corrcoef(sx, gx)[0,1]:+.3f}")
    print(f"  corr(start_y, goal_y) = {np.corrcoef(sy, gy)[0,1]:+.3f}")
    print(f"  circ-corr(start_th, goal_th) = {circ_corr(sth, gth):+.3f}")
    print(f"  relative rotation: mean|.| {np.abs(relrot).mean():.0f}deg  median|.| {np.median(np.abs(relrot)):.0f}  "
          f"std {relrot.std():.0f}  (spread => angle not fixed)")

    print("\n--- (b) REGION predictability (the operative 'language load-bearing' check) ---")
    sc = np.array([c[0] + REGION_NCELLS * c[1] for c in start_cell])
    gc = np.array([c[0] + REGION_NCELLS * c[1] for c in goal_cell])
    K = REGION_NCELLS * REGION_NCELLS
    marg = np.bincount(gc, minlength=K).astype(float)
    H_goal = entropy(marg)
    # conditional entropy H(goal|start)
    H_cond = 0.0
    for s in range(K):
        m = sc == s
        if m.sum() == 0: continue
        H_cond += (m.mean()) * entropy(np.bincount(gc[m], minlength=K).astype(float))
    p_leave = float((sc != gc).mean())
    print(f"  P(goal_cell != start_cell)      = {p_leave:.3f}  (goal usually leaves the start cell)")
    print(f"  H(goal_cell)                    = {H_goal:.2f} bits (max {np.log2(K):.2f})")
    print(f"  H(goal_cell | start_cell)       = {H_cond:.2f} bits")
    print(f"  info start gives about goal     = {H_goal - H_cond:.2f} bits "
          f"({100*(H_goal-H_cond)/max(H_goal,1e-9):.0f}% of goal entropy)")
    print(f"  goal-region marginal coverage   = {(marg>0).sum()}/{K} cells, "
          f"min/max share {marg.min()/n:.3f}/{marg.max()/n:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json_safe = {
            "n": n, "excluded": {"wall": n_wall, "short": n_short, "no_tf": n_nofit},
            "corr_x": float(np.corrcoef(sx, gx)[0,1]), "corr_y": float(np.corrcoef(sy, gy)[0,1]),
            "circ_corr_theta": circ_corr(sth, gth),
            "relrot_mean_abs": float(np.abs(relrot).mean()), "relrot_std": float(relrot.std()),
            "p_leave_cell": p_leave, "H_goal": H_goal, "H_goal_given_start": H_cond,
            "cells_covered": int((marg>0).sum()), "K": K,
        }
        import json; json.dump(json_safe, open(args.out, "w"), indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
