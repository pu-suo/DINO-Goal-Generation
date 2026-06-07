"""Local CPU smoke test for analysis/pose_decode_probe.py (no real data / GPU).

Synthesizes a tiny trajectory-latent cache whose patch latents LINEARLY encode the
block pose (so a correct probe must recover it with low error), then runs the probe
end-to-end and asserts: it loads the cache, the whole-traj split + masking + ridge +
MLP + smoothness + diagnosis run, the JSON/plots are written, and the masked MLP
orientation error on this clean signal is small (proves the angle/atan2/wrap math and
the masked-energy mask plumbing are wired correctly).

    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python analysis/_smoke_pose_decode.py
"""
import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
N_TOKENS, EMB, SIM = 196, 384, 512


def make_cache(root, split="train", n_traj=400, L=12, seed=0):
    rng = np.random.RandomState(seed)
    d = Path(root) / split
    d.mkdir(parents=True, exist_ok=True)
    # LOW-RANK signal (like real DINO): pose is encoded only in a few "object"
    # patches via a per-patch linear readout; the rest is constant + small noise.
    # This is rank-4 in pose, so a correct ridge/MLP recovers it from modest #frames
    # (a full-rank random projection into 75264 dims would need far more samples).
    n_obj = 16
    obj = rng.choice(N_TOKENS, n_obj, replace=False)
    A = (rng.randn(n_obj, EMB, 4) * 1.0).astype(np.float32)   # per-patch (EMB,4) readout
    const_bias = (rng.randn(N_TOKENS, EMB) * 0.3).astype(np.float32)

    lat_list, st_list, starts, lengths = [], [], [], []
    cursor = 0
    for t in range(n_traj):
        x0, y0 = rng.uniform(120, 392, 2)
        x1, y1 = rng.uniform(120, 392, 2)
        th0 = rng.uniform(0, 2 * np.pi)
        th1 = th0 + rng.uniform(-1.5, 1.5)          # smooth monotone-ish rotation
        ts = np.linspace(0, 1, L)
        bx = x0 + (x1 - x0) * ts
        by = y0 + (y1 - y0) * ts
        th = (th0 + (th1 - th0) * ts) % (2 * np.pi)
        # pusher trails the block, so its patches (and the mask) move over time
        ax = bx + rng.uniform(-30, 30)
        ay = by + rng.uniform(-30, 30)
        vx = rng.randn(L); vy = rng.randn(L)
        st = np.stack([ax, ay, bx, by, th, vx, vy], axis=1).astype(np.float32)
        pose4 = np.stack([bx / SIM, by / SIM, np.cos(th), np.sin(th)], axis=1).astype(np.float32)
        # background patches are CONSTANT across frames (-> standardize to ~0, no
        # interference); only the object patches carry the pose signal + tiny noise.
        lat = np.broadcast_to(const_bias, (L, N_TOKENS, EMB)).copy()
        sig = np.einsum("lk,pek->lpe", pose4, A)            # (L,n_obj,EMB)
        lat[:, obj, :] += sig + rng.randn(L, n_obj, EMB).astype(np.float32) * 0.02
        lat = lat.astype(np.float16)
        lat_list.append(torch.from_numpy(lat))
        st_list.append(torch.from_numpy(st))
        starts.append(cursor); lengths.append(L); cursor += L

    torch.save(torch.cat(lat_list, 0), d / "latents.pth")
    torch.save(torch.cat(st_list, 0), d / "states.pth")
    torch.save(torch.tensor(starts, dtype=torch.long), d / "traj_starts.pth")
    torch.save(torch.tensor(lengths, dtype=torch.long), d / "traj_lengths.pth")
    json.dump({"frameskip": 5, "n_traj": n_traj, "n_model_steps": cursor,
               "state_dim": 7, "latent_shape": [N_TOKENS, EMB]}, open(d / "meta.json", "w"))
    return cursor


def main():
    tmp = tempfile.mkdtemp(prefix="pose_decode_smoke_")
    out = os.path.join(tmp, "out")
    try:
        n = make_cache(tmp)
        print(f"synthetic cache: {n} model-steps at {tmp}")
        cmd = [sys.executable, os.path.join(HERE, "pose_decode_probe.py"),
               "--cache_dir", tmp, "--split", "train", "--device", "cpu",
               "--max_train_frames", "5000", "--max_test_frames", "2000",
               "--mlp_epochs", "100", "--min_smooth_len", "8", "--n_smooth_traj", "3",
               "--out", out]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(r.stdout[-3000:])
        if r.returncode != 0:
            print("STDERR:\n", r.stderr[-3000:])
            raise SystemExit("probe exited non-zero")

        rep = json.load(open(os.path.join(out, "pose_decode_probe.json")))
        mr, mm, diag = rep["masked"]["ridge"], rep["masked"]["mlp"], rep["diagnosis"]
        jr = diag["jitter_residual_deg"]
        print(f"\nSMOKE CHECK: masked LINEAR theta={mr['theta_mae_deg']:.1f}deg pos={mr['pos_l2_mae_px']:.1f}px"
              f" | masked MLP(pca) theta={mm['theta_mae_deg']:.1f}deg pos={mm['pos_l2_mae_px']:.1f}px"
              f" | resid_jitter={jr:.1f}deg | verdict={diag['verdict']}")
        for f in ("theta_scatter.png", "xy_scatter.png", "theta_time.png", "err_hist.png"):
            assert os.path.exists(os.path.join(out, f)), f"missing plot {f}"
        # The LINEAR probe must near-perfectly decode this clean low-rank signal -- the
        # definitive check that the probe math (pos L2, angle wrap/atan2, masking, traj
        # split, ridge intercept) is correct.
        assert mr["theta_mae_deg"] < 5, f"LINEAR theta {mr['theta_mae_deg']:.1f}deg -- math broken"
        assert mr["pos_l2_mae_px"] < 20, f"LINEAR pos_L2 {mr['pos_l2_mae_px']:.1f}px -- math broken"
        assert mr["frac_both"] > 0.9, f"LINEAR within-gate frac {mr['frac_both']:.2f} -- math broken"
        # Smoothness now uses the LINEAR decoder -> residual jitter must be ~0 on a clean
        # signal (the bug was the MLP fabricating jitter; this guards the fix).
        assert jr < 5, f"residual jitter {jr:.1f}deg -- smoothness decoder fabricating jitter"
        # MLP on PCA features must be well-conditioned now (no longer garbage on clean data).
        assert mm["theta_mae_deg"] < 20, f"MLP(pca) theta {mm['theta_mae_deg']:.1f}deg -- PCA-MLP broken"
        assert {"verdict", "driver", "linear_minus_mlp_gap_deg", "pusher_theta_gain_deg",
                "jitter_residual_deg", "jitter_decoded_deg"} <= set(diag)
        print("SMOKE OK: linear probe near-perfect (math correct); PCA-MLP well-conditioned; "
              "linear-decoder smoothness residual ~0; masking + diagnosis + 4 plots all run.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
