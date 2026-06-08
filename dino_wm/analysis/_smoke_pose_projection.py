"""Local CPU smoke for planning/pose_projection_cost.py (no GPU / real data / gym).

Validates the projection-metric planning cost end to end against the REAL interfaces:
  1. decode prep: a synthetic linear decoder reading pose from z[patch0, dims0:4] is recovered
     by the cost's flatten->standardize->@W+ymu (the same prep the probe persists).
  2. tolerance calibration: at the success boundary each normalized term ~= 1.0
     (20px -> pos term 1.0 ; 20deg -> ang term 1.0).
  3. pose ranking: pose-only cost (lambda_l2=0) ranks a closer-pose candidate below a far one.
  4. lambda_l2 dominance: with pose tied, large lambda_l2 ranks the lower background-L2 candidate first.
  5. masked-L2 EQUIVALENCE: the cost's lambda_l2 term == planning.objectives.objective_fn_last
     (alpha=0) under an arbitrary 0/1 vis_mask -- proves R4 (same masked-mean as the floor).
  6. Hydra: conf/cost/{masked_l2,pose_projection}.yaml both instantiate via hydra.utils.call.

    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python analysis/_smoke_pose_projection.py
"""
import os
import sys
import math
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from planning.pose_projection_cost import create_pose_projection_fn, N_TOKENS, EMB
from planning.objectives import create_objective_fn


def make_decoder(path, dilation=0):
    """Decoder that reads pose from z_flat[0:4] = z[patch0, dims 0:4]: identity W, no norm."""
    D = N_TOKENS * EMB
    W = torch.zeros(D, 4)
    for k in range(4):
        W[k, k] = 1.0
    torch.save({"mu": torch.zeros(D), "sd": torch.ones(D), "W": W, "ymu": torch.zeros(4),
                "n_tokens": N_TOKENS, "emb": EMB, "dilation": dilation, "masked": True,
                "pose_param": ["x_px", "y_px", "cos", "sin"], "prep": "smoke",
                "metrics": {"theta_mae_deg": 0.0}}, path)


def lat(x, y, th, bg=0.0, B=1):
    """(B,P,D) latent encoding pose in patch0 dims0:4; constant background `bg` in patch5."""
    z = torch.zeros(B, N_TOKENS, EMB)
    z[:, 0, 0], z[:, 0, 1] = x, y
    z[:, 0, 2], z[:, 0, 3] = math.cos(th), math.sin(th)
    z[:, 5, :] = bg
    return z


def pred(z):  # (B,P,D) -> rollout dict with a T axis (cost takes [:, -1])
    return {"visual": z.unsqueeze(1), "proprio": torch.zeros(z.shape[0], 1, 2)}


def tgt(z):
    return {"visual": z.unsqueeze(1), "proprio": torch.zeros(z.shape[0], 1, 2)}


def main():
    tmp = tempfile.mkdtemp(prefix="pose_proj_smoke_")
    ck = os.path.join(tmp, "linear_decoder.pt")
    make_decoder(ck)

    # 2. tolerance calibration (pose-only) ------------------------------------------------
    cost = create_pose_projection_fn(ck, w_pos=1.0, w_ang=1.0, lambda_l2=0.0)
    g = lat(256, 256, 0.0)
    c_pos20 = float(cost(pred(lat(276, 256, 0.0)), tgt(g))[0])          # 20px pos error
    c_ang20 = float(cost(pred(lat(256, 256, math.radians(20))), tgt(g))[0])  # 20deg ang error
    print(f"calibration: pos@20px term={c_pos20:.4f}  ang@20deg term={c_ang20:.4f}")
    assert abs(c_pos20 - 1.0) < 1e-3, f"pos tol term {c_pos20} != 1.0"
    assert abs(c_ang20 - 1.0) < 1e-3, f"ang tol term {c_ang20} != 1.0"

    # 3. pose ranking ---------------------------------------------------------------------
    c_near = float(cost(pred(lat(260, 256, math.radians(3))), tgt(g))[0])
    c_far = float(cost(pred(lat(360, 200, math.radians(90))), tgt(g))[0])
    assert c_near < c_far, f"pose-only must rank near<far ({c_near} !< {c_far})"

    # 4. lambda_l2 dominance (pose tied via identical patch0; differ in background patch5) -
    cost_l2 = create_pose_projection_fn(ck, w_pos=1.0, w_ang=1.0, lambda_l2=1e6)
    g_bg = lat(256, 256, 0.0, bg=1.0)
    c_samebg = float(cost_l2(pred(lat(256, 256, 0.0, bg=1.0)), tgt(g_bg))[0])  # bg matches goal
    c_diffbg = float(cost_l2(pred(lat(256, 256, 0.0, bg=5.0)), tgt(g_bg))[0])  # bg differs
    assert c_samebg < c_diffbg, f"large lambda_l2 must prefer lower bg-L2 ({c_samebg} !< {c_diffbg})"

    # 5. masked-L2 term == floor objective_fn_last(alpha=0) under a 0/1 mask ---------------
    torch.manual_seed(0)
    B = 7
    zp = torch.randn(B, N_TOKENS, EMB)
    zg = torch.randn(B, N_TOKENS, EMB)
    mask = (torch.rand(N_TOKENS) > 0.3).float()                      # arbitrary keep mask
    l2_only = create_pose_projection_fn(ck, w_pos=0.0, w_ang=0.0, lambda_l2=1.0,
                                        normalize_to_tol=False)
    floor = create_objective_fn(alpha=0, base=2, mode="last")
    a = l2_only(pred(zp), tgt(zg), vis_mask=mask)
    b = floor(pred(zp), tgt(zg), vis_mask=mask)
    assert a.shape == (B,) and b.shape == (B,), f"shape {a.shape} {b.shape}"
    assert torch.allclose(a, b, atol=1e-5), f"masked-L2 term != floor: max|d|={(a-b).abs().max():.2e}"
    # and unmasked too
    a0 = l2_only(pred(zp), tgt(zg))
    b0 = floor(pred(zp), tgt(zg))
    assert torch.allclose(a0, b0, atol=1e-5), "unmasked-L2 term != floor"

    # 6. both Hydra cost configs instantiate ----------------------------------------------
    import hydra
    from omegaconf import OmegaConf
    pp = OmegaConf.load(os.path.join(REPO, "conf/cost/pose_projection.yaml"))
    pp.decoder_ckpt = ck                                            # point at the smoke decoder
    fn_pp = hydra.utils.call(OmegaConf.to_container(pp, resolve=True))
    assert callable(fn_pp) and fn_pp(pred(g), tgt(g)).shape == (1,)
    ml = OmegaConf.load(os.path.join(REPO, "conf/cost/masked_l2.yaml"))
    fn_ml = hydra.utils.call(OmegaConf.to_container(ml, resolve=True))
    assert callable(fn_ml) and fn_ml(pred(zp), tgt(zg)).shape == (B,)

    print("SMOKE OK: decode prep correct; tolerance-calibrated (1.0 at gate); pose ranks; "
          "lambda_l2 dominates; masked-L2 == floor objective_fn_last(alpha=0); both cost YAMLs "
          "instantiate via hydra.")


if __name__ == "__main__":
    main()
