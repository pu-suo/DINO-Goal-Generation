"""Phase-3 Stage-0b: ADJUDICATE g's marginal pose-decode (is 20.8px real scatter or a
probe-transfer artifact?) and decompose the error into bias vs scatter, PER AXIS.

The Stage-0 number (real-fit decoder -> generated latents = 20.8px) is ambiguous: the
decoder was fit on REAL latents, so if g's latents are slightly off the real-latent
distribution but still LINEARLY encode the correct pose, the real-fit decoder extrapolates
poorly and OVERSTATES g's true error. Three decoders settle it (all masked, same protocol):

  real->real   : decoder fit on REAL train, eval REAL test goals      (scene ceiling ~9.5px)
  real->gen    : decoder fit on REAL train, eval GENERATED test goals (the ambiguous 20.8px)
  gen->gen     : decoder fit on GENERATED train, eval GENERATED test goals (THE ADJUDICATOR;
                 held-out split of g's own latents)

Reading (per the review):
  gen->gen ~= real->real  => pose info is linearly present in g's output at full fidelity;
                             20.8 was the real-fit probe failing to transfer -> BENIGN, lean GO.
  gen->gen still ~20px    => info genuinely absent below 20px -> real scatter, ceiling is real
                             -> GO-WITH-CHANGES (stronger synthesis / calibration / metric).

Plus a per-axis (x, y, theta) BIAS vs SCATTER decomposition on the generated goals:
  bias present (systematic offset / under-rotation) => CORRECTABLE by calibration.
  no bias, just scatter                             => domain-shift (if gen->gen good) or hard.
theta is reported separately and watched hardest (it has no patch-resolution excuse and is the
binding constraint for the 20deg gate); the right-skew tail is reported as p90|err|.

Box (4090):
  /workspace/envs/dino_wm/bin/python analysis/eval_coord_precision.py \
    --ckpt outputs/bridge/g_coord/g_best.pth \
    --latent_dir /workspace/data/pusht_coord/latents --data_path /workspace/data/pusht_coord \
    --out analysis_outputs/coord_precision.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bridge import BridgeG
from analysis.pose_decode_probe import (
    fit_linear, predict_linear, build_keep_masks, POS_TOL_PX, ANG_TOL_DEG,
)


def load_g(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["config"]
    g = BridgeG(dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"], cond_mode="coord",
                n_freq=cfg.get("n_freq", 12), heat_sigma=cfg.get("heat_sigma", 1.2)).to(device)
    g.load_state_dict(ck["state_dict"]); g.eval()
    return g, ck


def mask_flatten(z, agent_xy, dilation):
    sr = torch.cat([agent_xy, torch.zeros(len(agent_xy), 3)], dim=1)  # cols 0:2 = pusher xy
    keep = build_keep_masks(sr, dilation)
    return (z * keep[:, :, None]).reshape(len(z), -1)


def pose_Y(pose):
    return torch.stack([pose[:, 0], pose[:, 1], torch.cos(pose[:, 2]), torch.sin(pose[:, 2])], 1)


def signed_ang_err_deg(pred4, theta_true):
    th_pred = torch.atan2(pred4[:, 3], pred4[:, 2])
    e = th_pred - theta_true
    e = (e + np.pi) % (2 * np.pi) - np.pi        # signed, wrapped to [-pi,pi]
    return torch.rad2deg(e)


def report(pred4, pose):
    dx = pred4[:, 0] - pose[:, 0]
    dy = pred4[:, 1] - pose[:, 1]
    pos = torch.sqrt(dx ** 2 + dy ** 2)
    ang = signed_ang_err_deg(pred4, pose[:, 2])
    aabs = ang.abs()
    return {
        "pos_mae_px": float(pos.mean()), "pos_median_px": float(pos.median()),
        "ang_mae_deg": float(aabs.mean()), "ang_median_deg": float(aabs.median()),
        "ang_p90_deg": float(torch.quantile(aabs, 0.90)),
        "frac_pos_lt20": float((pos < POS_TOL_PX).float().mean()),
        "frac_ang_lt20": float((aabs < ANG_TOL_DEG).float().mean()),
        "frac_both": float(((pos < POS_TOL_PX) & (aabs < ANG_TOL_DEG)).float().mean()),
        # bias (mean signed err) vs scatter (std) per axis -> correctable bias if |bias|>>0
        "bias_x_px": float(dx.mean()), "scatter_x_px": float(dx.std()),
        "bias_y_px": float(dy.mean()), "scatter_y_px": float(dy.std()),
        "bias_theta_deg": float(ang.mean()), "scatter_theta_deg": float(ang.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out", default="analysis_outputs/coord_precision.json")
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--max_fit", type=int, default=6000)
    ap.add_argument("--dilation", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    def load_split(sp):
        ld, dp = Path(args.latent_dir) / sp, Path(args.data_path) / sp
        return {
            "z_start": torch.load(ld / "start_latents.pth").float(),
            "z_goal": torch.load(ld / "goal_latents.pth").float(),
            "spec": torch.load(ld / "specs.pth").float(),
            "start_pose": torch.load(ld / "start_poses.pth").float(),
            "agent_xy": torch.load(dp / "init_states.pth").float()[:, :2],
        }
    tr, te = load_split("train"), load_split("test")
    g, ck = load_g(args.ckpt, device)

    @torch.no_grad()
    def gen_goal(split):
        out = []
        for i in range(0, len(split["z_start"]), 512):
            zs = split["z_start"][i:i + 512].to(device)
            sp = split["spec"][i:i + 512].to(device)
            out.append(g.forward_coord(zs, sp).cpu())
        return torch.cat(out, 0)
    gtr, gte = gen_goal(tr), gen_goal(te)

    def subsample(n):
        if n <= args.max_fit:
            return torch.arange(n)
        gg = torch.Generator().manual_seed(0)
        return torch.randperm(n, generator=gg)[:args.max_fit]

    # --- decoder REAL: fit on real train (start+goal frames) ---
    zr = torch.cat([tr["z_start"], tr["z_goal"]], 0)
    axr = torch.cat([tr["agent_xy"], tr["agent_xy"]], 0)
    pr = torch.cat([tr["start_pose"], tr["spec"]], 0)
    sel = subsample(len(zr))
    Xr = mask_flatten(zr[sel], axr[sel], args.dilation)
    _, dec_real = fit_linear(Xr, pose_Y(pr[sel]), Xr[:1], args.ridge_lambda, device)

    # --- decoder GEN: fit on generated train goal latents (held-out from test) ---
    selg = subsample(len(gtr))
    Xg = mask_flatten(gtr[selg], tr["agent_xy"][selg], args.dilation)
    _, dec_gen = fit_linear(Xg, pose_Y(tr["spec"][selg]), Xg[:1], args.ridge_lambda, device)

    # --- eval on held-out TEST ---
    Xte_real = mask_flatten(te["z_goal"], te["agent_xy"], args.dilation)
    Xte_gen = mask_flatten(gte, te["agent_xy"], args.dilation)
    res = {
        "real_to_real": report(predict_linear(dec_real, Xte_real), te["spec"]),  # scene ceiling
        "real_to_gen":  report(predict_linear(dec_real, Xte_gen), te["spec"]),   # ambiguous 20.8
        "gen_to_gen":   report(predict_linear(dec_gen, Xte_gen), te["spec"]),    # ADJUDICATOR
    }
    out = {"n_test": len(te["z_goal"]), "ridge_lambda": args.ridge_lambda, "dilation": args.dilation,
           "max_fit": args.max_fit, "g_epoch": ck.get("epoch"),
           "g_val_changed_cos": ck.get("val_changed_cos"), **res}
    os.makedirs(Path(args.out).parent, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    def line(tag, d):
        return (f"{tag:11s} pos {d['pos_mae_px']:5.2f}px med {d['pos_median_px']:5.2f} | "
                f"ang {d['ang_mae_deg']:5.2f}deg med {d['ang_median_deg']:5.2f} p90 {d['ang_p90_deg']:5.1f} | "
                f"in-gate {d['frac_both']*100:4.1f}% (pos {d['frac_pos_lt20']*100:.0f}/ang {d['frac_ang_lt20']*100:.0f})")
    print("\n=== Stage-0b precision adjudication (n_test={}) ===".format(len(te["z_goal"])))
    print(line("real->real", res["real_to_real"]), " <- scene ceiling")
    print(line("real->gen ", res["real_to_gen"]),  " <- ambiguous (real-fit probe on g)")
    print(line("gen->gen  ", res["gen_to_gen"]),   " <- ADJUDICATOR")
    print("\nper-axis bias / scatter on GENERATED goals (gen->gen decoder):")
    d = res["gen_to_gen"]
    print(f"  x: bias {d['bias_x_px']:+.2f}px scatter {d['scatter_x_px']:.2f}px")
    print(f"  y: bias {d['bias_y_px']:+.2f}px scatter {d['scatter_y_px']:.2f}px")
    print(f"  theta: bias {d['bias_theta_deg']:+.2f}deg scatter {d['scatter_theta_deg']:.2f}deg  (WATCH THETA)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
