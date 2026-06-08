"""Projection-metric planning cost for the masked-actuator DINO-WM (docs/POSE_COST_SWEEP.md).

Wires the frozen LINEAR pose decoder from analysis/pose_decode_probe.py into the CEM cost as a
projection metric, mixed with the committed masked-L2 floor energy:

    C(z_T ; z_g) = w_pos*pos_scale*||p_hat - p_g||^2          # decoded block position (px^2)
                 + w_ang*ang_scale*||(c,s)_hat - (c,s)_g||^2  # decoded orientation (chordal^2)
                 + lambda_l2 * masked_mean_L2(z_T, z_g)       # the 0.80 floor term

DEPLOYABLE: uses only the rollout terminal latent z_T and the GOAL LATENT z_g. The goal pose is
DECODED from z_g inside the cost -- never passed as ground truth -- so the path is identical when
z_g = g(z_start, text) later. No real pusher, no GT pose.

SUPERVISED, NOT bootstrapped: the decoder is a frozen static readout fit by supervised regression
against ground-truth pose (the probe measured 4.4deg / 5.4px / ~3.2deg jitter). It is NOT a
TD/Bellman value head -- that was the failed quasimetric family (docs/RULED_OUT.md). Do not add a
bootstrapped target here.

The cost is a CLOSURE with the EXACT existing objective signature so it drops into the CEM loop
unchanged: objective_fn(z_obs_pred, z_obs_tgt, vis_mask=None) -> (B,)  [planning/cem.py:146].
"""
import math
from pathlib import Path

import torch

N_TOKENS, EMB = 196, 384


def _resolve_ckpt(decoder_ckpt):
    """Resolve decoder_ckpt against the repo root when relative.

    plan.py runs under @hydra.main, which chdir's into the run output dir, so a relative
    path would not resolve. parents[1] of this file == the repo root (dino_wm/).
    """
    p = Path(decoder_ckpt)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    if not p.exists():
        raise FileNotFoundError(
            f"pose decoder not found at {p}. Run analysis/pose_decode_probe.py first to "
            f"persist linear_decoder.pt (see docs/POSE_COST_SWEEP.md Step 1)."
        )
    return p


def create_pose_projection_fn(
    decoder_ckpt="analysis_outputs/pose_decode_probe/linear_decoder.pt",
    w_pos=1.0,
    w_ang=1.0,
    lambda_l2=0.0,
    pos_tol_px=20.0,
    ang_tol_deg=20.0,
    normalize_to_tol=True,
):
    """Return objective_fn(z_obs_pred, z_obs_tgt, vis_mask=None) -> (B,) loss.

    Args mirror docs/POSE_COST_SWEEP.md §2. `decoder_ckpt` is the masked linear decoder dumped
    by pose_decode_probe.py: {mu,sd: (196*384,); W: (196*384,4); ymu: (4,)} with decode
    [x_px, y_px, cos, sin] = ((z_flat - mu)/sd) @ W + ymu  (NO pooling; full flattened grid).
    """
    ck = torch.load(_resolve_ckpt(decoder_ckpt), map_location="cpu")
    D_in = N_TOKENS * EMB
    W = ck["W"].float()
    assert tuple(W.shape) == (D_in, 4), f"decoder W must be ({D_in},4), got {tuple(W.shape)}"
    mu = ck["mu"].float().reshape(-1)
    sd = ck["sd"].float().reshape(-1)
    ymu = ck["ymu"].float().reshape(-1)
    assert mu.numel() == D_in and sd.numel() == D_in and ymu.numel() == 4
    dec_dilation = int(ck.get("dilation", 0))

    if normalize_to_tol:
        pos_scale = (1.0 / float(pos_tol_px)) ** 2
        chord_tol = (2.0 * math.sin(math.radians(float(ang_tol_deg)) / 2.0)) ** 2  # ~0.1206 @20deg
        ang_scale = 1.0 / chord_tol
    else:
        pos_scale = ang_scale = 1.0

    print(f"[cost] pose_projection: w_pos={w_pos} w_ang={w_ang} lambda_l2={lambda_l2} "
          f"| pos_scale={pos_scale:.5f} ang_scale={ang_scale:.3f} (normalize_to_tol={normalize_to_tol}) "
          f"| decoder dilation={dec_dilation} metrics={ck.get('metrics', {}).get('theta_mae_deg','?')}deg")

    # decoder tensors are moved to the latent's device/dtype on first call and cached
    cache = {"device": None, "W": None, "mu": None, "sd": None, "ymu": None}
    state = {"warned_nomask": False}

    def _to(device, dtype):
        if cache["device"] != (device, dtype):
            cache.update(device=(device, dtype),
                         W=W.to(device=device, dtype=dtype),
                         mu=mu.to(device=device, dtype=dtype),
                         sd=sd.to(device=device, dtype=dtype),
                         ymu=ymu.to(device=device, dtype=dtype))
        return cache

    def _decode(z_masked):
        """z_masked: (B, P, D) already masked. -> (pos (B,2) px, unit (c,s) (B,2))."""
        c = cache
        zf = z_masked.reshape(z_masked.shape[0], -1)          # (B, P*D), row-major patch-major
        out = ((zf - c["mu"]) / c["sd"]) @ c["W"] + c["ymu"]  # (B, 4) = [x, y, cos, sin]
        pos = out[:, 0:2]
        cs = out[:, 2:4]
        cs = cs / cs.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # project to unit circle
        return pos, cs

    def objective_fn(z_obs_pred, z_obs_tgt, vis_mask=None):
        # single terminal frame on both sides (the dynamics needs history; the cost does not)
        zp = z_obs_pred["visual"][:, -1]                      # (B, P, D)  predicted terminal
        zg = z_obs_tgt["visual"][:, -1]                       # (B, P, D)  goal (T=1)
        assert zp.shape[-2:] == (N_TOKENS, EMB), f"unexpected latent shape {tuple(zp.shape)}"
        _to(zp.device, zp.dtype)

        # SAME mask on both sides, and it feeds the decoder too (one mask source). m in {0,1}.
        if vis_mask is not None:
            m = vis_mask.to(device=zp.device, dtype=zp.dtype).view(1, -1, 1)   # (1,P,1)
            zp_m, zg_m = zp * m, zg * m
        else:
            if not state["warned_nomask"]:
                print("[cost] pose_projection: vis_mask is None -> decoder (trained MASKED) sees "
                      "the unmasked latent. Pass mask_pusher=true for the deployable energy.")
                state["warned_nomask"] = True
            m, zp_m, zg_m = None, zp, zg

        # --- masked-L2 floor term (matches planning.objectives._masked_visual_mean) ---
        se = (zp_m - zg_m) ** 2                               # (B,P,D); m in {0,1} -> m^2 = m
        if m is not None:
            c_l2 = se.sum(dim=(1, 2)) / (m.sum() * EMB + 1e-8)
        else:
            c_l2 = se.mean(dim=(1, 2))

        # --- projection (pose) term: decode BOTH sides with the frozen probe ---
        p_pred, cs_pred = _decode(zp_m)
        p_goal, cs_goal = _decode(zg_m)
        c_pos = ((p_pred - p_goal) ** 2).sum(dim=-1)          # px^2
        c_ang = ((cs_pred - cs_goal) ** 2).sum(dim=-1)        # chordal^2 on the unit circle

        return (w_pos * pos_scale * c_pos
                + w_ang * ang_scale * c_ang
                + lambda_l2 * c_l2)

    return objective_fn
