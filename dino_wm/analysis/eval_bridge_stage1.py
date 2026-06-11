"""Stage-1 evaluation of the trained bridge `g` (specs/G_ARCHITECTURE.md §8, decoder-free).

Measures latent FIDELITY + GROUNDING on the HELD-OUT test combos, with the always-run ablations
(swapped-text, instruction-agnostic floor, retrieval baseline). No planning / dynamics model needed.

A. FIDELITY (the Stage-1 gate): g(z_start, instruction) vs the real enc(o_goal):
   - changed-region cosine (GATE >= 0.90, MICRO-averaged to match training), full-grid cosine + L2.
   - vs identity (z_start) and a nearest-neighbor RETRIEVAL baseline (a fidelity floor; the spec's
     off-manifold retrieval test is a Stage-2 planning comparison, not this changed-cos floor).
B. GROUNDING via the frozen pose decoder (linear_decoder.pt from pose_decode_probe; reused here):
   - first VALIDATE transfer: decode the REAL goal latent's pose vs the label goal_pose. If the
     decoder transfers to multicolor (decals present), trust it; else report fidelity only.
   - decode g's z_goal pose -> error (px/deg) vs the named target; frac within the 20px/20deg gate.
   - SWAPPED-TEXT: name a DIFFERENT visible target -> does g's decoded pose move to THAT target?
     (text load-bearing = an interpretable failure, per the spec).
   - INSTRUCTION-AGNOSTIC floor: a neutral instruction -> grounding should collapse.

Run (box, after train_bridge.py):
  python analysis/eval_bridge_stage1.py --ckpt outputs/bridge/g0/g_best.pth \
    --latent_dir $DATASET_DIR/pusht_multicolor/latents --data_path $DATASET_DIR/pusht_multicolor \
    --pose_decoder analysis_outputs/pose_decode_probe/linear_decoder.pt --split test
Local smoke: analysis/_smoke_eval_bridge.py (dummy text + synthetic decoder).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.pusht_multicolor_dset import PushTMultiColorLatentGoalDataset
from env.pusht.multicolor_common import manipulator_energy_mask, DEFAULT_PALETTE
from env.pusht.instructions import render_instruction
from models.bridge import BridgeG, changed_region_mask

POS_TOL_PX, ANG_TOL_DEG = 20.0, 20.0
_COLORS = [c for c, _ in DEFAULT_PALETTE]


# ----------------------------------------------------------------------------- model / text
def load_g(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    c = ck["config"]
    g = BridgeG(dim=c["dim"], depth=c["depth"], heads=c["heads"], d_text=c["d_text"]).to(device)
    g.load_state_dict(ck["state_dict"])
    g.eval()
    return g, ck.get("tau", None), ck.get("text_model", None), c.get("text_max_len", 16)


def build_text_encoder(text_model, dummy, width, device, text_max_len=16):
    if dummy or text_model is None:
        from train_bridge import DummyTextEncoder
        return DummyTextEncoder(d_text=width)
    from models.bridge import FrozenTextEncoder
    return FrozenTextEncoder(text_model, max_len=text_max_len, device=device)


def encode_texts(texts, encoder, device):
    tk, mk = encoder(list(texts))
    return tk.to(device), mk.to(device)


# ----------------------------------------------------------------------------- pose decode
def load_pose_decoder(path, device):
    ck = torch.load(path, map_location=device)
    return {"mu": ck["mu"].float().to(device), "sd": ck["sd"].float().to(device),
            "W": ck["W"].float().to(device), "ymu": ck["ymu"].float().to(device),
            "dilation": int(ck.get("dilation", 0))}


def decode_pose(z, pusher_xy, dec, device):
    """z (B,196,384), pusher_xy (B,2) sim-512 -> (x,y,theta). Masks pusher patches (as trained),
    flattens, applies the frozen linear decoder [x_px,y_px,cos,sin]."""
    masks = np.stack([manipulator_energy_mask([pusher_xy[i]], dilation=dec["dilation"])
                      for i in range(len(pusher_xy))])
    m = torch.from_numpy(masks).to(device).float()
    zf = (z * m[:, :, None]).reshape(z.shape[0], -1)
    out = ((zf - dec["mu"]) / dec["sd"]) @ dec["W"] + dec["ymu"]
    return out[:, 0], out[:, 1], torch.atan2(out[:, 3], out[:, 2])


def pose_err(px, py, pth, gx, gy, gth):
    pos = torch.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    d = torch.abs(pth - gth) % (2 * np.pi)
    ang = torch.rad2deg(torch.minimum(d, 2 * np.pi - d))
    return pos, ang


# ----------------------------------------------------------------------------- metrics
def fidelity(pred, target, z_start, tau):
    changed = changed_region_mask(z_start, target, tau)                       # (B,196)
    cos = F.cosine_similarity(pred, target, dim=-1)                           # (B,196)
    nchg = changed.sum(1)                                                     # (B,) changed-patch count
    cos_ch = (cos * changed).sum(1) / nchg.clamp_min(1.0)                     # per-sample changed-cos
    cos_full = cos.mean(1)
    l2 = (pred - target).pow(2).sum(-1).mean(1)
    return cos_ch, nchg, cos_full, l2


def micro_changed_cos(cos_ch, nchg):
    """Global (MICRO) changed-cos == the TRAIN metric (train_bridge.changed_region_cosine):
    sum over ALL changed patches, not a per-sample macro mean. Zero-changed samples drop out
    naturally (cos_ch*nchg = 0 and nchg = 0), so they can't poison the gate. Using the macro mean
    here would falsely fail the 0.90 gate on any test sample whose patches all sit below train-tau."""
    return float((cos_ch * nchg).sum() / nchg.sum().clamp_min(1.0))


def retrieval_goal(z_start_test, z_start_train, z_goal_train):
    """For each test start, nearest train start (L2) -> its goal latent (the retrieval baseline)."""
    d = torch.cdist(z_start_test.flatten(1), z_start_train.flatten(1))        # (Nte, Ntr)
    nn = d.argmin(1)
    return z_goal_train[nn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--train_split", default="train", help="retrieval pool")
    ap.add_argument("--pose_decoder", default="analysis_outputs/pose_decode_probe/linear_decoder.pt")
    ap.add_argument("--dummy_text", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="analysis_outputs/eval_bridge_stage1")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out, exist_ok=True)

    g, tau, text_model, text_max_len = load_g(args.ckpt, device)
    assert tau is not None, "ckpt missing 'tau' -- was it written by train_bridge.py?"
    encoder = build_text_encoder(text_model, args.dummy_text, g.dim, device, text_max_len)
    dset = PushTMultiColorLatentGoalDataset(args.latent_dir, args.data_path, args.split)
    n = len(dset)
    z_start = dset.start.to(device)
    z_goal = dset.goal.to(device)
    labels = dset.labels
    print(f"[eval] split={args.split} n={n} | tau={tau} | text_model={text_model or 'DUMMY'}")

    instr = [labels[i]["instruction"] for i in range(n)]
    with torch.no_grad():
        tk, mk = encode_texts(instr, encoder, device)
        z_pred = g(z_start, tk, mk)

    # --- A. fidelity (gate) ---  MICRO changed-cos == the train metric (matches train_bridge logging)
    cos_ch, nchg, cos_full, l2 = fidelity(z_pred, z_goal, z_start, tau)
    g_micro = micro_changed_cos(cos_ch, nchg)
    macro = float(cos_ch[nchg > 0].mean()) if bool((nchg > 0).any()) else float("nan")
    n_zero = int((nchg == 0).sum())
    id_cos_ch, id_nchg, _, _ = fidelity(z_start, z_goal, z_start, tau)         # identity baseline
    tr = PushTMultiColorLatentGoalDataset(args.latent_dir, args.data_path, args.train_split)
    z_retr = retrieval_goal(z_start, tr.start.to(device), tr.goal.to(device))
    re_cos_ch, re_nchg, _, _ = fidelity(z_retr, z_goal, z_start, tau)
    id_micro, re_micro = micro_changed_cos(id_cos_ch, id_nchg), micro_changed_cos(re_cos_ch, re_nchg)
    print("\n[A] LATENT FIDELITY (held-out):")
    print(f"  g            : changed-cos {g_micro:.4f} (micro; macro {macro:.4f})  "
          f"full-cos {cos_full.mean():.4f}  full-L2 {l2.mean():.2f}")
    print(f"  identity     : changed-cos {id_micro:.4f}   (z_start, no change)")
    print(f"  retrieval-NN : changed-cos {re_micro:.4f}   (nearest train goal -- fidelity floor, NOT the spec's off-manifold test)")
    if n_zero:
        print(f"  note: {n_zero}/{n} test samples have NO changed patches at train-tau (dropped from changed-cos)")
    gate_fid = g_micro >= 0.90

    out = {"split": args.split, "n": n, "tau": tau, "text_model": text_model,
           "fidelity": {"g_changed_cos": g_micro, "g_changed_cos_macro": macro, "n_zero_changed": n_zero,
                        "g_full_cos": float(cos_full.mean()), "g_full_l2": float(l2.mean()),
                        "identity_changed_cos": id_micro, "retrieval_changed_cos": re_micro}}

    # --- A2. text sensitivity (DECODER-FREE grounding): name a DIFFERENT color -> g's goal should
    # move AWAY from the named-target goal (cosine collapses toward the identity floor). This proves
    # text is load-bearing without depending on the pose decoder transferring to multicolor. ---
    active = np.array([int(labels[i]["active_idx"]) for i in range(n)])
    n_targets = len(labels[0]["target_colors"])
    swapped_idx = (active + 1) % n_targets
    sw_instr = [render_instruction(labels[i]["target_colors"][swapped_idx[i]],
                                   int(labels[i]["template_id"])) for i in range(n)]
    with torch.no_grad():
        stk, smk = encode_texts(sw_instr, encoder, device)
        z_sw = g(z_start, stk, smk)
    sw_cos_ch, sw_nchg, _, _ = fidelity(z_sw, z_goal, z_start, tau)            # swapped g vs NAMED goal
    sw_micro = micro_changed_cos(sw_cos_ch, sw_nchg)
    print("\n[A2] TEXT SENSITIVITY (decoder-free): cosine of g's goal to the NAMED-target goal:")
    print(f"  correct text {g_micro:.4f}  ->  swapped color {sw_micro:.4f}   (identity floor {id_micro:.4f})")
    print(f"  -> drop {g_micro - sw_micro:.3f}; swapped near the {id_micro:.3f} floor = text fully load-bearing")
    out["text_sensitivity"] = {"correct_changed_cos": g_micro, "swapped_changed_cos": sw_micro,
                               "identity_changed_cos": id_micro}

    # --- B. grounding via pose decoder (optional; needs a decoder that transfers to multicolor) ---
    if os.path.exists(args.pose_decoder):
        dec = load_pose_decoder(args.pose_decoder, device)
        pusher = np.stack([np.asarray(labels[i]["init_state"], dtype=np.float64)[:2] for i in range(n)])
        gp = np.stack([np.asarray(labels[i]["goal_pose"], dtype=np.float64) for i in range(n)])  # (N,3) x,y,theta
        gx = torch.tensor(gp[:, 0], device=device); gy = torch.tensor(gp[:, 1], device=device)
        gth = torch.tensor(gp[:, 2], device=device)

        # B0: decoder-transfer check on the REAL goal latent. The stock-pusht decoder is applied to
        # MULTICOLOR goals (decals present) -> must verify it still reads pose. Median alone passes on
        # a bad tail (where g's hard cases live); require MOST goals within gate + a p90 guard.
        rx, ry, rth = decode_pose(z_goal, pusher, dec, device)
        t_pos, t_ang = pose_err(rx, ry, rth, gx, gy, gth)
        t_within = float(((t_pos < POS_TOL_PX) & (t_ang < ANG_TOL_DEG)).float().mean())
        transfers = (t_within >= 0.80
                     and float(t_pos.quantile(0.9)) < 2 * POS_TOL_PX
                     and float(t_ang.quantile(0.9)) < 2 * ANG_TOL_DEG)
        print(f"\n[B0] pose-decoder transfer on REAL goal latent: within-gate {t_within:.2f} | "
              f"pos med {t_pos.median():.1f}/p90 {t_pos.quantile(0.9):.1f}px | "
              f"ang med {t_ang.median():.1f}/p90 {t_ang.quantile(0.9):.1f}deg -> "
              f"{'TRANSFERS' if transfers else 'WEAK -> grounding UNTRUSTED (use fidelity only)'}")

        # B1: grounding of g's z_goal vs the NAMED target
        px, py, pth = decode_pose(z_pred, pusher, dec, device)
        pos, ang = pose_err(px, py, pth, gx, gy, gth)
        within = ((pos < POS_TOL_PX) & (ang < ANG_TOL_DEG)).float().mean()
        print(f"[B1] g grounding vs NAMED target: pos {pos.mean():.1f}px (med {pos.median():.1f}) "
              f"ang {ang.mean():.1f}deg (med {ang.median():.1f}) | within-gate {within:.2f}")

        # B2: swapped-text -> does g move to the SWAPPED target? (reuses z_sw / swapped_idx from A2)
        tpos = np.stack([np.asarray(labels[i]["target_poses"], dtype=np.float64) for i in range(n)])  # (N,K,3)
        sx, sy, _ = decode_pose(z_sw, pusher, dec, device)
        sxy = torch.stack([sx, sy], 1)
        named_xy = torch.tensor(tpos[np.arange(n), active, :2], device=device)
        swap_xy = torch.tensor(tpos[np.arange(n), swapped_idx, :2], device=device)
        start_xy = torch.tensor(np.stack([np.asarray(labels[i]["init_state"], dtype=np.float64)[2:4]
                                          for i in range(n)]), device=device)  # block start
        d_named = (sxy - named_xy).norm(dim=1)
        d_swap = (sxy - swap_xy).norm(dim=1)
        d_start = (sxy - start_xy).norm(dim=1)
        follows_swapped = (d_swap < d_named).float().mean()
        # stricter: g actually MOVED the T to the swapped target (closer to swap than to the block's
        # START) -- rules out a near-identity g that copies z_start and scores ~0.5 by geometry alone.
        moved_to_swapped = ((d_swap < d_named) & (d_swap < d_start)).float().mean()
        print(f"[B2] SWAPPED-TEXT: closer-to-swapped {follows_swapped:.2f} | "
              f"MOVED-to-swapped (vs block start) {moved_to_swapped:.2f}  "
              f"(>=0.8 => text load-bearing, not identity)")

        # B3: instruction-agnostic floor (neutral text, no color)
        with torch.no_grad():
            ntk, nmk = encode_texts(["push the T to the target"] * n, encoder, device)
            z_neu = g(z_start, ntk, nmk)
        nx, ny, nth = decode_pose(z_neu, pusher, dec, device)
        n_pos, _ = pose_err(nx, ny, nth, gx, gy, gth)
        print(f"[B3] instruction-agnostic floor: pos {n_pos.mean():.1f}px vs named "
              f"(should be WORSE than g's {pos.mean():.1f}px if text matters)")

        out["grounding"] = {
            "decoder_trustworthy": bool(transfers), "transfer_within_gate": t_within,
            "transfer_pos_med_px": float(t_pos.median()), "transfer_ang_med_deg": float(t_ang.median()),
            "named_pos_mae_px": float(pos.mean()), "named_ang_mae_deg": float(ang.mean()),
            "named_within_gate": float(within),
            "swapped_follows_swapped": float(follows_swapped),
            "swapped_moved_to_swapped": float(moved_to_swapped),
            "agnostic_pos_mae_px": float(n_pos.mean())}
    else:
        print(f"\n[B] pose decoder not found at {args.pose_decoder} -> fidelity only "
              "(re-run the probe to persist linear_decoder.pt for grounding metrics)")

    # --- verdict ---
    print("\n" + "=" * 64)
    fid = out["fidelity"]
    L = [f"STAGE-1 VERDICT (split={args.split}):",
         f"  fidelity gate (changed-cos >= 0.90): {fid['g_changed_cos']:.3f} -> "
         f"{'PASS' if gate_fid else 'BELOW'}",
         f"  beats retrieval: {fid['g_changed_cos']:.3f} vs {fid['retrieval_changed_cos']:.3f} -> "
         f"{'yes' if fid['g_changed_cos'] > fid['retrieval_changed_cos'] else 'NO (g ~ nearest-neighbor)'}"]
    ts = out["text_sensitivity"]
    drop = ts["correct_changed_cos"] - ts["swapped_changed_cos"]
    # NOTE: a CORRECTLY-grounded g(swapped) shares the origin-erasure with the named goal and only
    # differs on the T-arrival patches, so it lands BETWEEN the floor and correct -- NOT at the floor.
    # The drop's SIGN confirms text is used; its magnitude is confounded by target geometry. The crisp
    # which-target grounding comes from a multicolor-refit pose decoder (B1/B2), not this number.
    L += [f"  text used (g(correct) {ts['correct_changed_cos']:.3f} -> g(swapped) {ts['swapped_changed_cos']:.3f}, "
          f"drop {drop:.3f} {'>0 yes' if drop > 0 else '<=0 NO'}); "
          f"crisp grounding needs a multicolor pose decoder (analysis/fit_multicolor_pose_decoder.py)"]
    if "grounding" in out:
        gr = out["grounding"]
        if gr["decoder_trustworthy"]:
            agn_ok = gr["agnostic_pos_mae_px"] > gr["named_pos_mae_px"]
            L += [f"  grounding within-gate: {gr['named_within_gate']:.2f}  "
                  f"(named pos {gr['named_pos_mae_px']:.1f}px / ang {gr['named_ang_mae_deg']:.1f}deg)",
                  f"  text load-bearing (moved-to-swapped): {gr['swapped_moved_to_swapped']:.2f} (>=0.8 good)",
                  f"  instruction-agnostic floor worse than g: {'yes' if agn_ok else 'NO (text may not matter!)'}"]
        else:
            L += ["  grounding UNTRUSTED: pose decoder did NOT transfer to multicolor (B0 WEAK) -> "
                  "rely on fidelity; refit the decoder on multicolor goals to grade pose grounding."]
    if text_model is None:
        L += ["  WARNING: DUMMY text encoder -> swapped/agnostic ablations are MEANINGLESS; "
              "re-run with real MiniLM (drop --dummy_text)."]
    L += ["  -> proceed to Stage-2 (CEM) only if fidelity PASSES and grounding is real (or deferred).", "=" * 64]
    print("\n".join(L))
    json.dump(out, open(Path(args.out) / f"stage1_{args.split}.json", "w"), indent=2)
    print(f"report -> {Path(args.out) / f'stage1_{args.split}.json'}")


if __name__ == "__main__":
    main()
