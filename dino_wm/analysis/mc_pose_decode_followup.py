"""Follow-ups to fit_multicolor_pose_decoder.py (the 32px/31deg @ n_fit=2000 result) -- adjudicate
fit-sample starvation vs a true representation blocker before calling multicolor pose-decodability
a Phase-1 blocker. Reference points: single-T probe = 5.4px/4.4deg/96%-in-gate at n_fit=16000
(analysis_outputs/pose_decode_probe/pose_decode_probe.json); multicolor goal refit = 32px/31deg/16%
at n_fit=2000.

  A) START-frame decode (block clear of decals)        -- is the difficulty decal-overlap-specific?
  B) GOAL-frame decode at n_fit in {500,1000,2000}     -- sample slope: starved vs plateaued
  C) fit 1600 train goals -> eval 400 train-combo vs 400 heldout-combo goals -- combo-shift cost
  D) fit starts+goals (4000) -> eval test goals        -- do generic pose samples help the goal fit?

Run (box):
  python analysis/mc_pose_decode_followup.py \
    --latent_dir $DATASET_DIR/pusht_multicolor/latents --data_path $DATASET_DIR/pusht_multicolor
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.pusht_multicolor_dset import PushTMultiColorLatentGoalDataset
from analysis.pose_decode_probe import fit_linear, wrapped_deg
from analysis.fit_multicolor_pose_decoder import masked_flat, pose4, pusher_xy, goal_pose

POS_TOL_PX, ANG_TOL_DEG = 20.0, 20.0


def start_pose(dset):
    """Block pose at t=0: init_state = [pusher_x, pusher_y, block_x, block_y, theta, vx, vy]."""
    return torch.tensor(np.stack([np.asarray(dset.labels[i]["init_state"], dtype=np.float64)[2:5]
                                  for i in range(len(dset))]), dtype=torch.float32)


def apply_dec(dec, X):
    return ((X - dec["mu"]) / dec["sd"]) @ dec["W"] + dec["ymu"]


def metrics(pred, Y):
    dx, dy = pred[:, 0] - Y[:, 0], pred[:, 1] - Y[:, 1]
    pos = torch.sqrt(dx ** 2 + dy ** 2)
    ang = wrapped_deg(torch.atan2(Y[:, 3], Y[:, 2]), pred[:, 2], pred[:, 3])
    return {"pos_mae_px": float(pos.mean()), "pos_med_px": float(pos.quantile(0.5)),
            "ang_mae_deg": float(ang.mean()), "ang_med_deg": float(ang.quantile(0.5)),
            "within_gate": float(((pos < POS_TOL_PX) & (ang < ANG_TOL_DEG)).float().mean())}


def show(tag, m):
    print(f"  [{tag:36s}] pos {m['pos_mae_px']:5.1f}px (med {m['pos_med_px']:5.1f})  "
          f"ang {m['ang_mae_deg']:5.1f}deg (med {m['ang_med_deg']:5.1f})  gate {m['within_gate']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dilation", type=int, default=0)
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="analysis_outputs/pose_decode_probe/mc_decode_followup.json")
    args = ap.parse_args()

    tr = PushTMultiColorLatentGoalDataset(args.latent_dir, args.data_path, "train")
    te = PushTMultiColorLatentGoalDataset(args.latent_dir, args.data_path, "test")
    pu_tr, pu_te = pusher_xy(tr), pusher_xy(te)
    Gtr, Gte = masked_flat(tr.goal, pu_tr, args.dilation), masked_flat(te.goal, pu_te, args.dilation)
    Str, Ste = masked_flat(tr.start, pu_tr, args.dilation), masked_flat(te.start, pu_te, args.dilation)
    Ygtr, Ygte = pose4(goal_pose(tr)), pose4(goal_pose(te))
    Ystr, Yste = pose4(start_pose(tr)), pose4(start_pose(te))
    out = {}

    print("== A) START-frame decode (block clear of decals) ==")
    pred, _ = fit_linear(Str, Ystr, Ste, args.ridge_lambda, args.device)
    out["A_start_2000"] = metrics(pred, Yste)
    show("start fit2000 -> test starts", out["A_start_2000"])

    print("== B) GOAL-frame decode, n_fit slope ==")
    rng = np.random.RandomState(0)
    for n in (500, 1000, 2000):
        sub = rng.choice(len(Gtr), n, replace=False) if n < len(Gtr) else np.arange(len(Gtr))
        pred, _ = fit_linear(Gtr[sub], Ygtr[sub], Gte, args.ridge_lambda, args.device)
        out[f"B_goal_{n}"] = metrics(pred, Ygte)
        show(f"goal fit{n} -> test goals", out[f"B_goal_{n}"])

    print("== C) combo-shift: fit 1600 train goals ==")
    perm = np.random.RandomState(1).permutation(len(Gtr))
    fit_i, ho_i = perm[:1600], perm[1600:]
    _, dec = fit_linear(Gtr[fit_i], Ygtr[fit_i], Gte[:1], args.ridge_lambda, args.device)
    out["C_train_combo"] = metrics(apply_dec(dec, Gtr[ho_i]), Ygtr[ho_i])
    out["C_heldout_combo"] = metrics(apply_dec(dec, Gte), Ygte)
    show("-> 400 train-combo goals", out["C_train_combo"])
    show("-> 400 heldout-combo goals", out["C_heldout_combo"])

    print("== D) starts+goals fit (4000) -> test goals ==")
    X, Y = torch.cat([Gtr, Str]), torch.cat([Ygtr, Ystr])
    pred, _ = fit_linear(X, Y, Gte, args.ridge_lambda, args.device)
    out["D_mixed_4000"] = metrics(pred, Ygte)
    show("mixed fit4000 -> test goals", out["D_mixed_4000"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
