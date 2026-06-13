"""Phase-3 Stage-0: does g's GENERATED z_goal decode to the spec pose at the
~4.4deg/5.4px class? (cheapest gate; fail fast before any planning.)

The pose-decode probe is fit on REAL latents, so two things must be checked, in order:
  (1) DECODER TRANSFER: fit the masked linear pose decoder on the coord scene's REAL
      latents (start frames @ start_pose + goal frames @ spec, with the known pusher xy),
      and confirm it decodes held-out REAL goal latents at the single-T class
      (~5.4px / ~4.4deg / ~96% in-gate). This also confirms the CLEAN coord scene is as
      pose-decodable as stock pusht_noise (the Phase-0 reference), adapted to the actual scene.
  (2) THE GATE: apply that SAME decoder to g's GENERATED z_goal = g(z_start, spec) and
      measure pos/ang error vs the spec. If generated decodes at the same class as real,
      g's latents are on-manifold for the probe -> proceed to Stage 1. If they decode much
      worse, synthesis is off-manifold -> STOP, do not spend planning compute.

Reuses analysis/pose_decode_probe.py utilities (dual-ridge fit, masked decode, pose math)
so the protocol is byte-identical to the single-T reference.

Box (4090):
  /workspace/envs/dino_wm/bin/python analysis/eval_coord_stage0.py \
    --ckpt outputs/bridge/g_coord/g_best.pth \
    --latent_dir /workspace/data/pusht_coord/latents \
    --data_path /workspace/data/pusht_coord \
    --out analysis_outputs/coord_stage0.json
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
    fit_linear, predict_linear, build_keep_masks, wrapped_deg,
    POS_TOL_PX, ANG_TOL_DEG, N_TOKENS,
)


def load_g(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["config"]
    g = BridgeG(dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"], cond_mode="coord",
                n_freq=cfg.get("n_freq", 12), heat_sigma=cfg.get("heat_sigma", 1.2)).to(device)
    g.load_state_dict(ck["state_dict"])
    g.eval()
    return g, ck


def states_rows(agent_xy, pose):
    """(N,5) = [ax, ay, bx, by, theta] as pose_decode_probe expects."""
    return torch.cat([agent_xy, pose[:, :2], pose[:, 2:3]], dim=1)


def decode_pose(z, agent_xy, dec, dilation):
    """Masked-decode (N,196,384) -> (N,4) [x,y,cos,sin] using the saved decoder protocol."""
    sr = torch.cat([agent_xy, torch.zeros(len(agent_xy), 3)], dim=1)  # only cols 0:2 used for mask
    keep = build_keep_masks(sr, dilation)                            # (N,196)
    Xf = (z * keep[:, :, None]).reshape(len(z), -1)
    return predict_linear(dec, Xf)


def pose_errors(pred4, pose):
    pos = torch.linalg.norm(pred4[:, :2] - pose[:, :2], dim=1)
    ang = wrapped_deg(pose[:, 2], pred4[:, 2], pred4[:, 3])
    return {
        "pos_mae_px": float(pos.mean()), "pos_median_px": float(pos.median()),
        "ang_mae_deg": float(ang.mean()), "ang_median_deg": float(ang.median()),
        "frac_pos_lt20": float((pos < POS_TOL_PX).float().mean()),
        "frac_ang_lt20": float((ang < ANG_TOL_DEG).float().mean()),
        "frac_both": float(((pos < POS_TOL_PX) & (ang < ANG_TOL_DEG)).float().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out", default="analysis_outputs/coord_stage0.json")
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--max_fit", type=int, default=6000, help="cap dual-ridge fit frames (O(n^2) kernel)")
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

    # ---- (1) fit the masked linear decoder on REAL coord latents (start + goal frames) ----
    # build (latent, pusher_xy, block_pose) over BOTH frame types of the train split
    z_fit = torch.cat([tr["z_start"], tr["z_goal"]], 0)
    ax_fit = torch.cat([tr["agent_xy"], tr["agent_xy"]], 0)
    pose_fit = torch.cat([tr["start_pose"], tr["spec"]], 0)
    if len(z_fit) > args.max_fit:                       # cap O(n^2) dual-ridge kernel
        g_ = torch.Generator().manual_seed(0)
        sel = torch.randperm(len(z_fit), generator=g_)[:args.max_fit]
        z_fit, ax_fit, pose_fit = z_fit[sel], ax_fit[sel], pose_fit[sel]
    sr_fit = states_rows(ax_fit, pose_fit)
    keep_fit = build_keep_masks(sr_fit, args.dilation)
    Xtr = (z_fit * keep_fit[:, :, None]).reshape(len(z_fit), -1)
    Ytr = torch.stack([pose_fit[:, 0], pose_fit[:, 1],
                       torch.cos(pose_fit[:, 2]), torch.sin(pose_fit[:, 2])], 1)
    # test real goal frames (held-out) for the decoder-transfer reference
    sr_te = states_rows(te["agent_xy"], te["spec"])
    keep_te = build_keep_masks(sr_te, args.dilation)
    Xte_realgoal = (te["z_goal"] * keep_te[:, :, None]).reshape(len(te["z_goal"]), -1)
    print(f"[fit] decoder on {len(z_fit)} real coord frames (start+goal); test {len(te['z_goal'])} real goal frames")
    _, dec = fit_linear(Xtr, Ytr, Xte_realgoal[:1], args.ridge_lambda, device)
    dec = {k: v for k, v in dec.items()}  # cpu

    real_goal = pose_errors(predict_linear(dec, Xte_realgoal), te["spec"])

    # ---- (2) THE GATE: g's GENERATED z_goal decoded by the SAME decoder ----
    g, ck = load_g(args.ckpt, device)
    with torch.no_grad():
        zg_gen = []
        for i in range(0, len(te["z_start"]), 512):
            zs = te["z_start"][i:i + 512].to(device)
            sp = te["spec"][i:i + 512].to(device)
            zg_gen.append(g.forward_coord(zs, sp).cpu())
        zg_gen = torch.cat(zg_gen, 0)
    gen_goal = pose_errors(decode_pose(zg_gen, te["agent_xy"], dec, args.dilation), te["spec"])

    result = {
        "n_test": len(te["z_goal"]), "ridge_lambda": args.ridge_lambda, "dilation": args.dilation,
        "g_epoch": ck.get("epoch"), "g_val_changed_cos": ck.get("val_changed_cos"),
        "decoder_on_REAL_goal": real_goal,        # (1) transfer reference: clean scene decodable?
        "decoder_on_GENERATED_goal": gen_goal,    # (2) the Stage-0 gate
        "single_T_reference": {"pos_mae_px": 5.4, "ang_mae_deg": 4.4, "frac_both": 0.96},
    }
    os.makedirs(Path(args.out).parent, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)

    def fmt(d):
        return (f"pos {d['pos_mae_px']:.2f}px (med {d['pos_median_px']:.2f}) | "
                f"ang {d['ang_mae_deg']:.2f}deg (med {d['ang_median_deg']:.2f}) | "
                f"in-gate {d['frac_both']*100:.1f}%")
    print("\n=== Stage-0 pose-decode ===")
    print(f"REAL goal latents  (decoder transfer, want ~5.4px/4.4deg/96%): {fmt(real_goal)}")
    print(f"GEN  goal latents  (THE GATE -- g on-manifold?):               {fmt(gen_goal)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
