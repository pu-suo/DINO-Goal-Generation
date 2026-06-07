"""
Phase-0 go/no-go probe for `g`: does the FROZEN, MASKED (object-only) DINO-WM
latent actually CONTAIN the block's pose (x, y, theta) that `g` must synthesize
and the CEM planner must reach?

WHY THIS IS THE DECIDER
-----------------------
The learned quasimetric cost-to-go did NOT beat the masked-L2 floor
(floor SR=0.80, qm SR=0.73, n=30); its gates were globally-monotone but
locally-noisy. Two hypotheses explain that:
  (A) SEARCH/VALUE failure  -- the pose info is in the latent, the value head
      just couldn't fit a clean cost-to-go. -> `g` is viable on THIS latent.
  (B) REPRESENTATION ceiling -- the masked DINO latent itself cannot resolve
      fine orientation; its own pose noise is larger than the success gate.
      -> no value function (and no `g`) can win here; a higher-resolution
      representation (e.g. V-JEPA-2-AC) is on the critical path BEFORE `g`.
This probe distinguishes (A) from (B) by directly DECODING block pose from the
frozen latent with a linear map and a small MLP, and comparing the decode error
to the success tolerances (pos < 20 sim-px, angle < pi/9 = 20 deg).

WHAT IT MEASURES (all on cached pusht_noise latents -- no re-encoding)
  MAIN     : decode pose from the MASKED object-only latent (pusher patches zeroed
             with the SAME manipulator_energy_mask the planner's energy uses).
  CONTROL 2: decode the SAME pose from the FULL UNMASKED latent; report
             (unmasked_err - masked_err) = how much the pusher carried the signal.
  CONTROL 3: SMOOTHNESS -- decode theta over time on held-out trajectories and
             report the frame-to-frame jitter (std of first-differences) of the
             decoded theta vs ground truth. Direct test of the "jittery fine-pose"
             hypothesis the quasimetric inherited.

DIAGNOSIS THRESHOLDS (printed against the 20px / 20deg gate; key off MASKED MLP)
  orientation MAE < ~15 deg  -> pose IS present; bottleneck is search/value, not
                                the representation; `g` is viable on this latent.
  orientation MAE > ~30-40deg -> hard representational ceiling; the masked DINO
                                latent cannot resolve fine orientation; a higher-
                                resolution representation is on the critical path.
  ~15-25 deg                 -> borderline: enough for the loose 20deg tolerance
                                on most goals, jittery on hard rotations (consistent
                                with the observed 0.80).
  If orientation MAE > 20deg -> the representation's OWN pose noise exceeds the
                                success criterion: no planner can reliably hit the
                                target regardless of the value function.
  If pusher-contribution large for orientation -> the pusher was the pose anchor.

DATA / SPLIT
  Reuses the cached full-trajectory model-step latents from pusht_noise (the cache
  the quasimetric used). Cache layout (per split, see cache_traj_latents.py):
    latents.pth (Ntot,196,384) f16 UNMASKED ; states.pth (Ntot,7) f32 RAW sim-512
    [ax,ay,bx,by,theta,vx,vy] -> pusher xy=[0:2], block pose=[2:5];
    traj_starts.pth / traj_lengths.pth -> per-traj row slices ; meta.json.
  SPLIT BY TRAJECTORY (whole trajs held out) -- never by frame (adjacent frames
  correlate and a frame split overstates decodability).

RUN (vast.ai, after the trajectory-latent cache exists)
  cd dino_wm && source $WS/activate.sh
  # default cache dir is .../pusht_noise/traj_latents; to reuse the EXISTING qm
  # cache from the quasimetric run without re-caching, point at qm_latents:
  python analysis/pose_decode_probe.py --cache_dir $DATASET_DIR/pusht_noise/qm_latents
Mac smoke (cpu, synthetic cache): see analysis/_smoke_pose_decode.py.
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow `python analysis/pose_decode_probe.py` from the repo root (dino_wm/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.pusht.multicolor_common import manipulator_energy_mask  # KEPT module

SIM = 512                 # sim/pygame frame; block (x,y) and the gate share this frame
POS_TOL_PX = 20.0         # stock PushT pose gate: ||block_xy_err|| < 20 sim-px
ANG_TOL_DEG = 20.0        # stock PushT pose gate: wrapped |dtheta| < pi/9 = 20 deg
N_TOKENS, EMB = 196, 384


# ----------------------------------------------------------------------------- io
def load_cache(cache_dir, split):
    d = Path(cache_dir) / split
    for f in ("latents.pth", "states.pth", "traj_starts.pth", "traj_lengths.pth"):
        if not (d / f).exists():
            raise SystemExit(f"missing {d/f} -- run scripts/cache_traj_latents.py "
                             f"--splits {split}  (or point --cache_dir at an existing cache)")
    try:                                         # mmap keeps the ~GB latent tensor off RAM
        latents = torch.load(d / "latents.pth", map_location="cpu", mmap=True)
    except TypeError:                            # older torch w/o mmap kwarg
        latents = torch.load(d / "latents.pth", map_location="cpu")
    states = torch.load(d / "states.pth", map_location="cpu").float()
    starts = torch.load(d / "traj_starts.pth").long()
    lengths = torch.load(d / "traj_lengths.pth").long()
    meta = {}
    if (d / "meta.json").exists():
        meta = json.load(open(d / "meta.json"))
    assert latents.shape[1:] == (N_TOKENS, EMB), f"unexpected latent shape {tuple(latents.shape)}"
    assert states.shape[1] >= 5, f"states need >=5 cols (got {states.shape[1]})"
    return latents, states, starts, lengths, meta


def traj_split(starts, lengths, test_frac, seed):
    """Whole-trajectory split -> (train_frame_idx, test_frame_idx, test_traj_slices)."""
    n = len(lengths)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test_t = set(perm[:n_test].tolist())
    tr, te, te_slices = [], [], []
    for t in range(n):
        s, L = int(starts[t]), int(lengths[t])
        rows = list(range(s, s + L))
        if t in test_t:
            te += rows
            te_slices.append((s, L))
        else:
            tr += rows
    return np.array(tr), np.array(te), te_slices


def subsample(idx, cap, seed):
    if cap is None or len(idx) <= cap:
        return idx
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(idx, cap, replace=False))


# -------------------------------------------------------------------------- masks
def build_keep_masks(states_rows, dilation):
    """(M,196) keep mask per frame via the planner's manipulator_energy_mask.

    Memoized on rounded pusher xy (the mask is patch-quantized, so integer-rounded
    pusher positions collide heavily -> a few hundred unique masks for 10k+ frames).
    """
    px = states_rows[:, 0:2].numpy()
    cache, out = {}, np.empty((len(px), N_TOKENS), dtype=np.float32)
    for i in range(len(px)):
        k = (int(round(px[i, 0])), int(round(px[i, 1])))
        m = cache.get(k)
        if m is None:
            m = manipulator_energy_mask([px[i]], dilation=dilation)
            cache[k] = m
        out[i] = m
    return torch.from_numpy(out)


def gather_latents(latents, idx):
    """Materialize rows idx (from a possibly-mmap'd f16 tensor) as f32 (M,196,384)."""
    return latents[torch.as_tensor(idx)].float()


# ------------------------------------------------------------------- pose targets
def pose_targets(states_rows):
    x, y, th = states_rows[:, 2], states_rows[:, 3], states_rows[:, 4]
    return torch.stack([x, y, torch.cos(th), torch.sin(th)], dim=1)  # (M,4)


def wrapped_deg(theta_true, cos_p, sin_p):
    th_pred = torch.atan2(sin_p, cos_p)
    d = torch.abs(theta_true - th_pred) % (2 * np.pi)
    d = torch.minimum(d, 2 * np.pi - d)         # C1 T-shape -> full [0,2pi), only 2pi wrap
    return torch.rad2deg(d)


# ---------------------------------------------------------------------- regressors
def ridge_dual(Xtr, Ytr, Xte, lam, device):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(device)
    Xte = ((Xte - mu) / sd).to(device)
    ymu = Ytr.mean(0, keepdim=True).to(device)        # intercept: center Y (x,y are ~256, not 0)
    Ytr = Ytr.to(device) - ymu
    K = Xtr @ Xtr.T
    A = K + lam * torch.eye(K.shape[0], device=device)
    alpha = torch.linalg.solve(A, Ytr)
    return ((Xte @ Xtr.T) @ alpha + ymu).cpu()


def train_mlp(Xtr, Ytr, Xte, epochs, lr, wd, hidden, device, dropout=0.1):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(device)
    Xte = ((Xte - mu) / sd).to(device)
    scale = torch.tensor([SIM, SIM, 1.0, 1.0])           # x,y -> ~[0,1]; cos,sin O(1)
    Ytr_n = (Ytr / scale).to(device)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(), nn.Dropout(dropout),
                        nn.Linear(hidden, 4)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    bs = min(256, Xtr.shape[0])
    for _ in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); lossf(net(Xtr[idx]), Ytr_n[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = (net(Xte).cpu() * scale)
    return pred, (net, mu, sd, scale)


# -------------------------------------------------------------------------- report
def score(name, pred, pose_te):
    # pose_te is the 4-col target [x, y, cos(theta), sin(theta)]; recover true angle.
    dx = pred[:, 0] - pose_te[:, 0]
    dy = pred[:, 1] - pose_te[:, 1]
    pos_l2 = torch.sqrt(dx ** 2 + dy ** 2)                # 2D Euclidean = GATE metric
    theta_true = torch.atan2(pose_te[:, 3], pose_te[:, 2])
    ang = wrapped_deg(theta_true, pred[:, 2], pred[:, 3])
    m = {
        "x_mae_px": float(dx.abs().mean()), "y_mae_px": float(dy.abs().mean()),
        "pos_l2_mae_px": float(pos_l2.mean()), "pos_l2_median_px": float(pos_l2.median()),
        "theta_mae_deg": float(ang.mean()), "theta_median_deg": float(ang.median()),
        "frac_pos_lt20": float((pos_l2 < POS_TOL_PX).float().mean()),
        "frac_ang_lt20": float((ang < ANG_TOL_DEG).float().mean()),
        "frac_both": float(((pos_l2 < POS_TOL_PX) & (ang < ANG_TOL_DEG)).float().mean()),
    }
    print(f"  {name:16s}: pos_L2={m['pos_l2_mae_px']:5.1f}px (med {m['pos_l2_median_px']:4.1f}) "
          f"theta={m['theta_mae_deg']:5.1f}deg (med {m['theta_median_deg']:4.1f}) "
          f"| <20px {m['frac_pos_lt20']:.2f}  <20deg {m['frac_ang_lt20']:.2f}  both {m['frac_both']:.2f}")
    return m, pos_l2, ang


# ------------------------------------------------------------------------ smoothness
def smoothness(model, latents, states, te_slices, dilation, n_traj, min_len, device, masked):
    """Decode theta over time on held-out trajs; jitter = std of first-diffs."""
    net, mu, sd, scale = model
    rows = []
    chosen = [(s, L) for (s, L) in te_slices if L >= min_len][:n_traj]
    for (s, L) in chosen:
        idx = np.arange(s, s + L)
        z = gather_latents(latents, idx)                       # (L,196,384)
        st = states[idx]
        if masked:
            keep = build_keep_masks(st, dilation)
            z = z * keep[:, :, None]
        X = z.reshape(L, -1)
        X = ((X - mu) / sd).to(device)
        with torch.no_grad():
            pred = (net(X).cpu() * scale)
        th_dec = torch.atan2(pred[:, 3], pred[:, 2]).numpy()   # decoded theta (wrapped)
        th_true = st[:, 4].numpy()
        # unwrap before diffing so a +/-2pi seam isn't counted as jitter
        d_dec = np.diff(np.unwrap(th_dec))
        d_true = np.diff(np.unwrap(th_true))
        rows.append({"start": int(s), "len": int(L),
                     "jit_dec_deg": float(np.rad2deg(np.std(d_dec))),
                     "jit_true_deg": float(np.rad2deg(np.std(d_true))),
                     "th_dec": th_dec, "th_true": th_true})
    return rows


# ----------------------------------------------------------------------- diagnosis
def diagnose(masked_mlp, masked_ridge, unmasked_mlp, unmasked_ridge, jit_rows):
    mlp_ang = masked_mlp["theta_mae_deg"]
    ridge_ang = masked_ridge["theta_mae_deg"]
    # "is the pose recoverable" -> use the BEST decoder (a latent either contains the
    # info or it doesn't; the tighter of linear/MLP is the honest decodability bound,
    # and this is robust to one probe under-training). Report both + the gap.
    ang = min(mlp_ang, ridge_ang)
    driver = "MLP" if mlp_ang <= ridge_ang else "linear"
    drv_masked = masked_mlp if driver == "MLP" else masked_ridge
    drv_unmasked = unmasked_mlp if driver == "MLP" else unmasked_ridge
    ang_med = drv_masked["theta_median_deg"]
    lin_mlp_gap = ridge_ang - mlp_ang          # >0: MLP beats linear (entangled); <0: MLP undertrained
    pos = min(masked_mlp["pos_l2_mae_px"], masked_ridge["pos_l2_mae_px"])
    pos_frac = max(masked_mlp["frac_pos_lt20"], masked_ridge["frac_pos_lt20"])
    # control 2 on the DRIVER probe: <0 means unmasked decodes orientation better
    pusher_gain = drv_unmasked["theta_mae_deg"] - drv_masked["theta_mae_deg"]
    jit_dec = float(np.mean([r["jit_dec_deg"] for r in jit_rows])) if jit_rows else float("nan")
    jit_true = float(np.mean([r["jit_true_deg"] for r in jit_rows])) if jit_rows else float("nan")

    L = ["", "=" * 72,
         f"DIAGNOSIS  (masked object-only latent; orientation MAE: linear {ridge_ang:.1f}deg, "
         f"MLP {mlp_ang:.1f}deg -> using {driver} {ang:.1f}deg)",
         "=" * 72]
    # --- orientation verdict ---
    if ang < 15:
        L += [f"ORIENTATION: {ang:.1f}deg < 15  -> POSE INFO IS PRESENT.",
              "  The masked DINO latent resolves orientation well within the 20deg gate.",
              "  The quasimetric's failure was SEARCH/VALUE-fitting, not the representation.",
              "  => `g` is VIABLE on this latent. Proceed to Phase 1 (build g)."]
        verdict = "g_viable"
    elif ang > 30:
        L += [f"ORIENTATION: {ang:.1f}deg > 30  -> HARD REPRESENTATIONAL CEILING.",
              "  The masked DINO latent cannot resolve fine orientation. No value function",
              "  and no `g` can reliably hit a 20deg target from this representation.",
              "  => A higher-resolution representation (e.g. V-JEPA-2-AC) is on the CRITICAL",
              "     PATH BEFORE building `g`."]
        verdict = "representation_ceiling"
    else:
        L += [f"ORIENTATION: {ang:.1f}deg in [15,30] -> BORDERLINE.",
              "  Enough for the loose 20deg tolerance on most goals, but jittery on hard",
              "  rotations -- consistent with the observed oracle SR~0.80.",
              "  => `g` is plausibly viable; expect a soft orientation ceiling."]
        verdict = "borderline"
    # --- explicit 'noise > gate' callout ---
    if ang > ANG_TOL_DEG:
        L += [f"  !! ORIENTATION MAE ({ang:.1f}deg) EXCEEDS the success tolerance (20deg):",
              "     the representation's OWN pose noise is larger than the success criterion,",
              "     so no planner can reliably hit the target regardless of the value function."]
    else:
        L += [f"  orientation MAE ({ang:.1f}deg) is within the 20deg gate (median {ang_med:.1f}deg)."]
    # --- accessibility: linear vs MLP gap ---
    if lin_mlp_gap > 5:
        L += [f"ACCESS: MLP beats linear by {lin_mlp_gap:.1f}deg -> pose is PRESENT but "
              "NONLINEARLY ENTANGLED (a linear read-out, like `g`'s simplest head, is insufficient)."]
    elif lin_mlp_gap < -5:
        L += [f"ACCESS: linear beats MLP by {(-lin_mlp_gap):.1f}deg -> the MLP appears UNDERTRAINED "
              "(it should match the linear probe at worst); trust the linear number and/or raise --mlp_epochs."]
    else:
        L += [f"ACCESS: linear and MLP agree ({ridge_ang:.1f} vs {mlp_ang:.1f}deg) -> pose is "
              "CLEANLY (linearly) ACCESSIBLE in the masked latent."]
    # --- position ---
    pos_tag = "within" if pos < POS_TOL_PX else "EXCEEDS"
    L += [f"POSITION: best pos_L2 MAE {pos:.1f}px {pos_tag} the 20px gate "
          f"(<20px on {pos_frac*100:.0f}% of frames)."]
    # --- pusher contribution (control 2) ---
    if pusher_gain < -10:
        L += [f"PUSHER: {driver} unmasked theta MAE is {(-pusher_gain):.1f}deg LOWER than masked "
              f"(unmasked {drv_unmasked['theta_mae_deg']:.1f} vs masked {drv_masked['theta_mae_deg']:.1f}).",
              "  The PUSHER was carrying much of the pose/search signal -- masking it out",
              "  (as the planner must, not knowing the goal-time pusher) removes that anchor."]
    else:
        L += [f"PUSHER: {driver} unmasked-masked theta MAE gap = {pusher_gain:+.1f}deg (small) -> the "
              "object-only latent carries the pose on its own; the pusher was not the anchor."]
    # --- smoothness (control 3) ---
    L += [f"SMOOTHNESS: decoded-theta frame-to-frame jitter {jit_dec:.1f}deg vs ground-truth "
          f"{jit_true:.1f}deg (held-out trajs).",
          ("  Decoded theta is much noisier than the true motion -> 'jittery fine-pose' confirmed."
           if jit_dec > jit_true + 3 else
           "  Decoded theta tracks the true motion smoothly -> no pathological jitter.")]
    L += ["=" * 72]
    print("\n".join(L))
    return {"verdict": verdict, "driver": driver,
            "masked_theta_mae_deg": ang, "masked_theta_median_deg": ang_med,
            "masked_theta_mae_mlp_deg": mlp_ang, "masked_theta_mae_linear_deg": ridge_ang,
            "linear_minus_mlp_gap_deg": lin_mlp_gap,
            "masked_pos_l2_mae_px": pos, "pusher_theta_gain_deg": pusher_gain,
            "jitter_decoded_deg": jit_dec, "jitter_true_deg": jit_true,
            "summary": "\n".join(L)}


# ---------------------------------------------------------------------------- plots
def save_plots(out_dir, preds, pose_te, pos_ang, jit_rows):
    out = Path(out_dir)
    # 1) theta scatter: {masked,unmasked} x {ridge,mlp}
    th_true_deg = np.rad2deg(np.arctan2(pose_te[:, 3].numpy(), pose_te[:, 2].numpy()))
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    for a, key in zip(ax.flat, ["masked_ridge", "masked_mlp", "unmasked_ridge", "unmasked_mlp"]):
        p = preds[key]
        tp = torch.atan2(p[:, 3], p[:, 2])
        a.scatter(th_true_deg, np.rad2deg(tp), s=6, alpha=0.4)
        a.plot([-180, 180], [-180, 180], "r--", lw=1)
        a.set_xlabel("theta true (deg)"); a.set_ylabel("theta pred (deg)"); a.set_title(key)
    fig.tight_layout(); fig.savefig(out / "theta_scatter.png", dpi=110); plt.close(fig)

    # 2) xy scatter (masked mlp)
    p = preds["masked_mlp"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    for a, j, lab in [(ax[0], 0, "x"), (ax[1], 1, "y")]:
        a.scatter(pose_te[:, j], p[:, j], s=6, alpha=0.4)
        a.plot([0, SIM], [0, SIM], "r--", lw=1)
        a.set_xlabel(f"{lab} true (px)"); a.set_ylabel(f"{lab} pred (px)"); a.set_title(f"masked_mlp {lab}")
    fig.tight_layout(); fig.savefig(out / "xy_scatter.png", dpi=110); plt.close(fig)

    # 3) theta vs time (masked mlp), sample held-out trajs
    if jit_rows:
        k = len(jit_rows)
        fig, ax = plt.subplots(1, k, figsize=(3.2 * k, 3.2), squeeze=False)
        for a, r in zip(ax[0], jit_rows):
            a.plot(np.rad2deg(np.unwrap(r["th_true"])), label="true", lw=1.5)
            a.plot(np.rad2deg(np.unwrap(r["th_dec"])), label="decoded", lw=1, alpha=0.8)
            a.set_xlabel("model step"); a.set_ylabel("theta (deg, unwrapped)")
            a.set_title(f"traj@{r['start']} jit {r['jit_dec_deg']:.0f}/{r['jit_true_deg']:.0f}")
        ax[0][0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "theta_time.png", dpi=110); plt.close(fig)

    # 4) error histograms (masked mlp) with gate lines
    pos_l2, ang = pos_ang
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].hist(pos_l2.numpy(), bins=40); ax[0].axvline(POS_TOL_PX, color="r", ls="--")
    ax[0].set_xlabel("pos L2 err (px)"); ax[0].set_title("masked_mlp position err")
    ax[1].hist(ang.numpy(), bins=40); ax[1].axvline(ANG_TOL_DEG, color="r", ls="--")
    ax[1].set_xlabel("theta err (deg)"); ax[1].set_title("masked_mlp orientation err")
    fig.tight_layout(); fig.savefig(out / "err_hist.png", dpi=110); plt.close(fig)


# ------------------------------------------------------------------------------ main
def run_condition(name, Xtr_full, ridge_idx, pose_tr, Xte, pose_te, args, device):
    """ridge on the capped subset (kernel cost), MLP once on the full train set."""
    print(f"[{name}] decode block pose (lower is better):")
    rp = ridge_dual(Xtr_full[ridge_idx], pose_tr[ridge_idx], Xte, args.ridge_lambda, device)
    rm, _, _ = score(f"{name}/ridge", rp, pose_te)
    mp, model = train_mlp(Xtr_full, pose_tr, Xte, args.mlp_epochs, args.mlp_lr, args.mlp_wd,
                          args.mlp_hidden, device)
    mm, pos_l2, ang = score(f"{name}/mlp  ", mp, pose_te)
    return {"ridge": rm, "mlp": mm}, {"ridge": rp, "mlp": mp}, model, (pos_l2, ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir",
                    default=os.path.join(os.environ.get("DATASET_DIR", "data"),
                                         "pusht_noise", "traj_latents"),
                    help="dir with <split>/{latents,states,traj_starts,traj_lengths}.pth "
                         "(reuse the quasimetric cache by pointing at .../pusht_noise/qm_latents)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--dilation", type=int, default=0, help="mask dilation (floor uses 0)")
    ap.add_argument("--max_train_frames", type=int, default=16000)
    ap.add_argument("--max_test_frames", type=int, default=8000)
    ap.add_argument("--ridge_max", type=int, default=6000, help="cap train rows for the ridge kernel")
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--mlp_epochs", type=int, default=200)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_wd", type=float, default=1e-4)
    ap.add_argument("--mlp_hidden", type=int, default=256)
    ap.add_argument("--n_smooth_traj", type=int, default=6)
    ap.add_argument("--min_smooth_len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis_outputs/pose_decode_probe")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out, exist_ok=True)

    latents, states, starts, lengths, meta = load_cache(args.cache_dir, args.split)
    print(f"cache: {latents.shape[0]} model-steps over {len(lengths)} trajs "
          f"(frameskip={meta.get('frameskip','?')}) | device={device}")

    tr_idx, te_idx, te_slices = traj_split(starts, lengths, args.test_frac, args.seed)
    tr_idx = subsample(tr_idx, args.max_train_frames, args.seed)
    te_idx = subsample(te_idx, args.max_test_frames, args.seed)
    print(f"split: {len(tr_idx)} train frames / {len(te_idx)} test frames "
          f"(whole-trajectory split, test_frac={args.test_frac})")

    # gather raw latents + targets once (mmap -> only these rows hit RAM)
    z_tr = gather_latents(latents, tr_idx)            # (Ntr,196,384) f32
    z_te = gather_latents(latents, te_idx)
    pose_tr = pose_targets(states[torch.as_tensor(tr_idx)])
    pose_te = pose_targets(states[torch.as_tensor(te_idx)])
    keep_tr = build_keep_masks(states[torch.as_tensor(tr_idx)], args.dilation)
    keep_te = build_keep_masks(states[torch.as_tensor(te_idx)], args.dilation)
    print(f"mask: avg kept patches/frame = {keep_tr.mean(0).sum():.0f}/196 (train)")

    ridge_tr = subsample(np.arange(len(tr_idx)), args.ridge_max, args.seed)

    out = {"split": args.split, "cache_dir": args.cache_dir, "device": device,
           "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
           "pos_tol_px": POS_TOL_PX, "ang_tol_deg": ANG_TOL_DEG, "meta": meta}
    preds_all, metrics, models = {}, {}, {}
    pos_ang_masked = None
    for name, masked in [("masked", True), ("unmasked", False)]:
        Xtr_full = (z_tr * keep_tr[:, :, None]) if masked else z_tr
        Xte = (z_te * keep_te[:, :, None]) if masked else z_te
        Xtr_full = Xtr_full.reshape(len(tr_idx), -1)
        Xte = Xte.reshape(len(te_idx), -1)
        m, p, model, pa = run_condition(
            name, Xtr_full, ridge_tr, pose_tr, Xte, pose_te, args, device)
        metrics[name], models[name] = m, model
        preds_all[f"{name}_ridge"], preds_all[f"{name}_mlp"] = p["ridge"], p["mlp"]
        if masked:
            pos_ang_masked = pa
        del Xtr_full, Xte

    print("\n[control 3] smoothness on held-out trajectories (masked):")
    jit = smoothness(models["masked"], latents, states, te_slices, args.dilation,
                     args.n_smooth_traj, args.min_smooth_len, device, masked=True)
    for r in jit:
        print(f"  traj@{r['start']:>6} L={r['len']:>2}: jitter decoded {r['jit_dec_deg']:5.1f}deg "
              f"vs true {r['jit_true_deg']:5.1f}deg")

    diag = diagnose(metrics["masked"]["mlp"], metrics["masked"]["ridge"],
                    metrics["unmasked"]["mlp"], metrics["unmasked"]["ridge"], jit)

    out["masked"], out["unmasked"] = metrics["masked"], metrics["unmasked"]
    out["smoothness"] = [{k: r[k] for k in ("start", "len", "jit_dec_deg", "jit_true_deg")} for r in jit]
    out["diagnosis"] = diag
    json.dump(out, open(Path(args.out) / "pose_decode_probe.json", "w"), indent=2)
    save_plots(args.out, preds_all, pose_te, pos_ang_masked, jit)
    print(f"\nreport -> {Path(args.out)/'pose_decode_probe.json'}  ; plots -> {args.out}/*.png")


if __name__ == "__main__":
    main()
