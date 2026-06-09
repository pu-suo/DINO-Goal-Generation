"""Refit the linear pose decoder ON MULTICOLOR goal latents (the stock-pusht decoder is OOD on the
colored decals -> ~98px/50deg, untrustworthy). Trains a linear map from the masked multicolor GOAL
latent to the block goal_pose [x,y,cos,sin], using the cached goal latents + labels. Saves in the
SAME format as analysis_outputs/pose_decode_probe/linear_decoder.pt, so eval_bridge_stage1.py
--pose_decoder <this> grades crisp px/deg grounding + the which-target swapped test.

Run (box):
  python analysis/fit_multicolor_pose_decoder.py \
    --latent_dir $DATASET_DIR/pusht_multicolor/latents --data_path $DATASET_DIR/pusht_multicolor \
    --out analysis_outputs/pose_decode_probe/multicolor_pose_decoder.pt
Local smoke: analysis/_smoke_fit_mc_decoder.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.pusht_multicolor_dset import PushTMultiColorLatentGoalDataset
from env.pusht.multicolor_common import manipulator_energy_mask
from analysis.pose_decode_probe import fit_linear, wrapped_deg

POS_TOL_PX, ANG_TOL_DEG = 20.0, 20.0


def masked_flat(z, pusher, dilation):
    """(N,196,384) goal latent, masked at the goal-frame pusher (=init/start pusher), flattened."""
    masks = np.stack([manipulator_energy_mask([pusher[i]], dilation=dilation) for i in range(len(pusher))])
    m = torch.from_numpy(masks).float()
    return (z.float() * m[:, :, None]).reshape(z.shape[0], -1)


def pose4(gp):  # (N,3) x,y,theta -> (N,4) x,y,cos,sin
    th = gp[:, 2]
    return torch.stack([gp[:, 0], gp[:, 1], torch.cos(th), torch.sin(th)], dim=1)


def pusher_xy(dset):
    return np.stack([np.asarray(dset.labels[i]["init_state"], dtype=np.float64)[:2] for i in range(len(dset))])


def goal_pose(dset):
    return torch.tensor(np.stack([np.asarray(dset.labels[i]["goal_pose"], dtype=np.float64)
                                  for i in range(len(dset))]), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dilation", type=int, default=0)
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="analysis_outputs/pose_decode_probe/multicolor_pose_decoder.pt")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tr = PushTMultiColorLatentGoalDataset(args.latent_dir, args.data_path, "train")
    te = PushTMultiColorLatentGoalDataset(args.latent_dir, args.data_path, "test")
    Xtr, Xte = masked_flat(tr.goal, pusher_xy(tr), args.dilation), masked_flat(te.goal, pusher_xy(te), args.dilation)
    Ytr, Yte = pose4(goal_pose(tr)), pose4(goal_pose(te))
    print(f"fit on {len(tr)} train goals, eval on {len(te)} test goals (masked at goal-frame pusher)")

    pred, dec = fit_linear(Xtr, Ytr, Xte, args.ridge_lambda, device)
    dx, dy = pred[:, 0] - Yte[:, 0], pred[:, 1] - Yte[:, 1]
    pos = torch.sqrt(dx ** 2 + dy ** 2)
    th_true = torch.atan2(Yte[:, 3], Yte[:, 2])
    ang = wrapped_deg(th_true, pred[:, 2], pred[:, 3])
    within = ((pos < POS_TOL_PX) & (ang < ANG_TOL_DEG)).float().mean()
    metrics = {"pos_l2_mae_px": float(pos.mean()), "pos_l2_median_px": float(pos.quantile(0.5)),
               "theta_mae_deg": float(ang.mean()), "theta_median_deg": float(ang.quantile(0.5)),
               "frac_within_gate": float(within)}
    print(f"[multicolor decoder] test: pos {metrics['pos_l2_mae_px']:.1f}px (med {metrics['pos_l2_median_px']:.1f}) "
          f"theta {metrics['theta_mae_deg']:.1f}deg (med {metrics['theta_median_deg']:.1f}) "
          f"| within-gate {within:.2f}")
    transfers = metrics["pos_l2_median_px"] < POS_TOL_PX and metrics["theta_median_deg"] < ANG_TOL_DEG
    print(f"  -> {'USABLE for grounding' if within >= 0.5 else 'still weak (block vs decals hard to separate linearly)'}")

    torch.save({"mu": dec["mu"].squeeze(0).contiguous(), "sd": dec["sd"].squeeze(0).contiguous(),
                "W": dec["W"].contiguous(), "ymu": dec["ymu"].squeeze(0).contiguous(),
                "n_tokens": 196, "emb": 384, "dilation": args.dilation, "masked": True,
                "pose_param": ["x_px", "y_px", "cos", "sin"],
                "prep": "mask goal-frame pusher (init_state[:2]) -> flatten 196*384 -> (x-mu)/sd -> @W+ymu",
                "trained_on": "multicolor goal latents", "metrics": metrics}, args.out)
    print(f"saved -> {args.out}  (use: eval_bridge_stage1.py --pose_decoder {args.out})")


if __name__ == "__main__":
    main()
