"""
Phase 0.4 - dynamics reuse-vs-retrain check.

Runs the SHIPPED PushT dynamics (frozen) on multi-color trajectories and measures
per-patch L2 latent-prediction error, decomposed by region (block / pusher /
target-marker / background) using the known poses. Two regimes:
  * teacher-forced 1-step  (predict next from TRUE history)
  * free H-step rollout    (predict from the true first num_hist frames only)

Decision rule (printed): REUSE if marker patches are copied stably (low error, no
drift) and block+pusher error is close to the single-target baseline; else RETRAIN
(-> 0.6). Pass --baseline_pusht_noise to also run on pusht_noise for that baseline.

Actions/proprio are normalized with PUSHT stats (what the shipped dynamics was
trained on), NOT the multi-color dataset's own stats.

    cd dino_wm
    DATASET_DIR=/data python analysis/dynamics_check.py \
        model_name=pusht ckpt_base_path=/ckpts --split test --n_traj 50
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

# allow `python analysis/dynamics_check.py` from the repo root
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from datasets.img_transforms import default_transform
from datasets.pusht_multicolor_dset import PushTMultiColorDataset
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD
from analysis.probe_common import patch_region_labels, REGION_NAMES
from utils import move_to_device


def normalize(obs_visual, act_raw, proprio_raw, frameskip, device):
    """Subsample-by-frameskip already done. Returns model-ready (1,S,...) tensors."""
    S = obs_visual.shape[0]
    proprio = (proprio_raw - PROPRIO_MEAN) / PROPRIO_STD
    act = act_raw.reshape(S, frameskip, 2)
    act = (act - ACTION_MEAN) / ACTION_STD
    act = rearrange(act, "s f d -> s (f d)")
    obs = {"visual": obs_visual.unsqueeze(0), "proprio": proprio.unsqueeze(0).float()}
    return move_to_device(obs, device), act.unsqueeze(0).float().to(device)


@torch.no_grad()
def per_patch_l2(z_pred, z_true):
    """z_*: (1, F, 196, 384) -> (F, 196) L2 over the feature dim."""
    return torch.linalg.norm(z_pred - z_true, dim=-1)[0].cpu().numpy()


def accumulate(err, region_labels, acc):
    """err: (F,196); region_labels: (F,196) -> add into acc[region] lists."""
    for ri, name in enumerate(REGION_NAMES):
        m = region_labels == ri
        if m.any():
            acc[name].append(float(err[m].mean()))


def run_dataset(wm, dset, model_cfg, n_traj, horizon, has_targets, device, seed=0):
    rng = np.random.RandomState(seed)
    nh, fs = model_cfg.num_hist, model_cfg.frameskip
    idxs = rng.choice(len(dset), min(n_traj, len(dset)), replace=False)
    tf_acc = {k: [] for k in REGION_NAMES}
    fr_acc = {k: [] for k in REGION_NAMES}

    for idx in idxs:
        obs, act, state, env_info = dset.get_frames(int(idx), range(dset.get_seq_length(int(idx))))
        T = obs["visual"].shape[0]
        S = T // fs
        if S < nh + 1:
            continue
        vis = obs["visual"][: S * fs : fs]                       # (S,3,H,W)
        st = state[: S * fs : fs].numpy()                        # (S,7)
        proprio_raw = torch.tensor(st[:, [0, 1, 5, 6]])
        act_raw = act[: S * fs].reshape(S, fs, 2)                # raw a (rel/scale)
        obs_in, act_n = normalize(vis, act_raw, proprio_raw, fs, device)

        z_true = wm.encode_obs(obs_in)["visual"]                 # (1,S,196,384)
        z_full = wm.encode(obs_in, act_n)                        # (1,S,196,dim_concat)

        target_poses = [t["pose"] for t in env_info["targets"]] if has_targets else []
        region = np.stack([
            patch_region_labels(st[i, 2:5], st[i, :2], target_poses) for i in range(S)
        ])  # (S,196)

        # teacher-forced 1-step
        for w in range(nh - 1, S - 1):
            zin = z_full[:, w - nh + 1: w + 1]
            z_pred = wm.predict(zin)
            pred_vis = wm.separate_emb(z_pred)[0]["visual"][:, -1:]   # (1,1,196,384)
            err = per_patch_l2(pred_vis, z_true[:, w + 1: w + 2])  # (1,196)
            accumulate(err, region[w + 1][None], tf_acc)           # both 2-D (1,196)

        # free rollout over horizon H from the first nh frames
        H = min(horizon, S - nh)
        if H >= 1:
            obs_0 = {"visual": obs_in["visual"][:, :nh], "proprio": obs_in["proprio"][:, :nh]}
            z_obses, _ = wm.rollout(obs_0, act_n[:, : nh + H])
            z_pred_free = z_obses["visual"][:, nh: nh + H]
            err = per_patch_l2(z_pred_free, z_true[:, nh: nh + H])    # (H,196)
            accumulate(err, region[nh: nh + H], fr_acc)

    summ = lambda acc: {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
    return {"teacher_forced_1step": summ(tf_acc), "free_rollout": summ(fr_acc),
            "n_traj": len(idxs), "horizon": horizon}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", nargs="?", default=None, help="(hydra-style kv also accepted)")
    ap.add_argument("--model_name", dest="model_name_kw", default="pusht")
    ap.add_argument("--ckpt_base_path", default="./checkpoints")
    ap.add_argument("--data_path", default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_multicolor"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_traj", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--baseline_pusht_noise", default=None, help="path to pusht_noise/val for the single-target baseline")
    ap.add_argument("--out", default="analysis_outputs/dynamics_check.json")
    args, extra = ap.parse_known_args()
    # accept hydra-style "model_name=pusht ckpt_base_path=/x"
    model_name = args.model_name_kw
    ckpt = args.ckpt_base_path
    for tok in ([args.model_name] if args.model_name else []) + extra:
        if tok and "=" in tok:
            k, v = tok.split("=", 1)
            if k == "model_name": model_name = v
            elif k == "ckpt_base_path": ckpt = v

    from plan import load_model  # lazy (pulls submitit/wandb)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = f"{ckpt}/outputs/{model_name}/"
    model_cfg = OmegaConf.load(os.path.join(model_path, "hydra.yaml"))
    wm = load_model(Path(model_path) / "checkpoints" / "model_latest.pth",
                    model_cfg, model_cfg.num_action_repeat, device=device)
    wm.eval()

    tfm = default_transform(model_cfg.img_size)
    dset = PushTMultiColorDataset(data_path=os.path.join(args.data_path, args.split),
                                  transform=tfm, normalize_action=False, with_velocity=True)
    report = {"multicolor_" + args.split: run_dataset(wm, dset, model_cfg, args.n_traj, args.horizon, True, device)}

    if args.baseline_pusht_noise:
        from datasets.pusht_dset import PushTDataset
        base = PushTDataset(data_path=args.baseline_pusht_noise, transform=tfm,
                            normalize_action=False, with_velocity=True)
        # PushTDataset.get_frames returns env_info={'shape':...}; wrap to no targets
        class _NoTgt:
            def __init__(s, d): s.d = d
            def __len__(s): return len(s.d)
            def get_seq_length(s, i): return s.d.get_seq_length(i)
            def get_frames(s, i, fr):
                o, a, st, _ = s.d.get_frames(i, fr); return o, a, st, {"targets": []}
        report["pusht_noise_baseline"] = run_dataset(wm, _NoTgt(base), model_cfg, args.n_traj, args.horizon, False, device)

    # decision heuristic.
    # Two questions that actually matter for REUSING the frozen dynamics:
    #   (1) do the colored decals DRIFT under rollout? -> measured as the marker
    #       region's rollout-growth RELATIVE TO background, not its absolute error
    #       (sharp colored edges have high latent error even when copied perfectly,
    #       so an absolute target/bg ratio mostly reflects edge-difficulty, not drift).
    #   (2) do the decals hurt prediction of the things we PLAN over (block, pusher)?
    #       -> block+pusher free-rollout error vs the single-target pusht_noise
    #       baseline under the SAME (pusht) normalization (controls for action scale).
    tf = report["multicolor_" + args.split]["teacher_forced_1step"]
    fr = report["multicolor_" + args.split]["free_rollout"]
    decision = "RETRAIN"
    reasons = []

    marker_stable = False
    if all(fr[r] is not None and tf[r] is not None for r in ("target", "background")):
        tgt_growth = fr["target"] / (tf["target"] + 1e-6)
        bg_growth = fr["background"] / (tf["background"] + 1e-6)
        drift = tgt_growth / (bg_growth + 1e-6)
        reasons.append(
            f"marker drift under rollout = {drift:.2f} "
            f"(decals grow {tgt_growth:.2f}x vs background {bg_growth:.2f}x; ~1 => copied stably)")
        marker_stable = drift < 1.3

    planning_ok = None
    if "pusht_noise_baseline" in report:
        base_bp = np.mean([report["pusht_noise_baseline"]["free_rollout"][r] for r in ("block", "pusher")])
        mc_bp = np.mean([fr[r] for r in ("block", "pusher")])
        ratio = mc_bp / (base_bp + 1e-6)
        reasons.append(
            f"block+pusher free-rollout err vs single-target baseline = {ratio:.2f} "
            f"(<=1.2 => decals don't degrade manipulation prediction)")
        planning_ok = ratio <= 1.2

    if marker_stable and planning_ok:
        decision = "REUSE"
    elif marker_stable and planning_ok is None:
        decision = "REUSE (provisional; add --baseline_pusht_noise to confirm block/pusher)"
    elif planning_ok and not marker_stable:
        decision = "REUSE-WITH-CAUTION (block/pusher fine but markers drift; check goal-marker latents)"
    report["decision"] = decision
    report["reasons"] = reasons

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\n--> DECISION: {decision}  (report -> {args.out})")


if __name__ == "__main__":
    main()
