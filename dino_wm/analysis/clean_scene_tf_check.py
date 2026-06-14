"""Part 3.2: clean-scene (green-T removed) OOD check for the FROZEN dynamics.

The dynamics was trained on frames WITH the fixed LightGreen goal-T at center. The
rigid pipeline renders CLEAN (green-T-removed) frames, so the transition model sees
an out-of-distribution scene. We re-render REAL pusht_noise trajectories from their
states in BOTH variants, teacher-force the frozen dynamics 1-step, and compare the
BLOCK-region latent error:

  green re-render  -> must reproduce the in-dist reference (~8.0) -> validates that
                      our state->render->encode pipeline matches the training data.
  clean re-render  -> the OOD number. If ~= green, removing the green-T is harmless
                      (it was inert/constant); if >>, a light dynamics retrain on
                      clean frames is needed before trusting the oracle.

  DATASET_DIR=/workspace/data python analysis/clean_scene_tf_check.py \
    model_name=pusht ckpt_base_path=/workspace/ckpts --n_traj 30
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from datasets.img_transforms import default_transform
from datasets.pusht_dset import PushTDataset, ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD
from datasets.rigid_goal_render import make_env, render_state
from analysis.probe_common import patch_region_labels
from utils import move_to_device

BLOCK = 1  # REGION_NAMES = [background, block, pusher, target]


@torch.no_grad()
def per_patch_l2(z_pred, z_true):
    return torch.linalg.norm(z_pred - z_true, dim=-1)[0].cpu().numpy()


def render_window(env, states5, tfm):
    imgs = np.stack([render_state(env, s)[0] for s in states5])      # (S,224,224,3) uint8
    x = torch.tensor(imgs).float() / 255.0
    x = rearrange(x, "s h w c -> s c h w")
    return tfm(x)                                                    # (S,3,H,W)


def normalize(vis, act_raw, proprio_raw, fs, device):
    S = vis.shape[0]
    proprio = (proprio_raw - PROPRIO_MEAN) / PROPRIO_STD
    act = ((act_raw.reshape(S, fs, 2) - ACTION_MEAN) / ACTION_STD)
    act = rearrange(act, "s f d -> s (f d)")
    obs = {"visual": vis.unsqueeze(0), "proprio": proprio.unsqueeze(0).float()}
    return move_to_device(obs, device), act.unsqueeze(0).float().to(device)


@torch.no_grad()
def block_err(wm, env, states5_full, states7, act, nh, fs, tfm, device):
    S = states7.shape[0] // fs          # subsampled frame count (states7 is S*fs long)
    vis = render_window(env, states5_full[: S * fs : fs], tfm)
    st = states7[: S * fs : fs]
    proprio_raw = st[:, [0, 1, 5, 6]]
    act_raw = act[: S * fs]
    obs_in, act_n = normalize(vis, act_raw, proprio_raw, fs, device)
    z_true = wm.encode_obs(obs_in)["visual"]
    z_full = wm.encode(obs_in, act_n)
    errs = []
    for w in range(nh - 1, S - 1):
        z_pred = wm.predict(z_full[:, w - nh + 1: w + 1])
        pred_vis = wm.separate_emb(z_pred)[0]["visual"][:, -1:]
        e = per_patch_l2(pred_vis, z_true[:, w + 1: w + 2])[0]       # (196,)
        lbl = patch_region_labels(st[w + 1, 2:5].numpy(), st[w + 1, :2].numpy(), [])
        m = lbl == BLOCK
        if m.any():
            errs.append(float(e[m].mean()))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", nargs="?", default=None)
    ap.add_argument("--ckpt_base_path", default="./checkpoints")
    ap.add_argument("--data_path", default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_noise"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_traj", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis_outputs/clean_scene_tf.json")
    args, extra = ap.parse_known_args()
    mn = "pusht"; ckpt = args.ckpt_base_path
    for tok in ([args.model_name] if args.model_name else []) + extra:
        if tok and "=" in tok:
            k, v = tok.split("=", 1)
            if k == "model_name": mn = v
            elif k == "ckpt_base_path": ckpt = v

    from plan import load_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mp = f"{ckpt}/outputs/{mn}/"
    model_cfg = OmegaConf.load(os.path.join(mp, "hydra.yaml"))
    wm = load_model(Path(mp) / "checkpoints" / "model_latest.pth", model_cfg,
                    model_cfg.num_action_repeat, device=device); wm.eval()
    nh, fs = model_cfg.num_hist, model_cfg.frameskip
    tfm = default_transform(model_cfg.img_size)

    dset = PushTDataset(data_path=os.path.join(args.data_path, args.split),
                        transform=tfm, normalize_action=False, with_velocity=True)
    env_green = make_env(with_target=True)
    env_clean = make_env(with_target=False)

    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(dset), min(args.n_traj, len(dset)), replace=False)
    green, clean = [], []
    for idx in idxs:
        L = int(dset.get_seq_length(int(idx)))
        S = min(nh + args.horizon + 1, L // fs)
        if S < nh + 1:
            continue
        states7 = dset.states[int(idx), : S * fs]
        states5 = states7[:, :5].numpy()
        act = dset.actions[int(idx), : S * fs]
        green += block_err(wm, env_green, states5, states7[:, :].clone(), act, nh, fs, tfm, device)
        clean += block_err(wm, env_clean, states5, states7[:, :].clone(), act, nh, fs, tfm, device)

    res = {"n_traj": len(idxs), "n_steps": len(green),
           "block_tf_green": float(np.mean(green)), "block_tf_green_std": float(np.std(green)),
           "block_tf_clean": float(np.mean(clean)), "block_tf_clean_std": float(np.std(clean)),
           "ratio_clean_over_green": float(np.mean(clean) / (np.mean(green) + 1e-9)),
           "reference_indist": 8.0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("\n=== Part 3.2 clean-scene TF (block-region 1-step latent L2) ===")
    print(f"  green re-render (self-check vs ~8.0): {res['block_tf_green']:.2f} +- {res['block_tf_green_std']:.2f}")
    print(f"  clean re-render (OOD question):       {res['block_tf_clean']:.2f} +- {res['block_tf_clean_std']:.2f}")
    print(f"  clean/green ratio: {res['ratio_clean_over_green']:.3f}  (~1 => removal harmless)")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
