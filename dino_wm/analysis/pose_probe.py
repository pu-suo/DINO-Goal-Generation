"""
Phase 0.3 pose-resolution probe: how finely does the 14x14 DINOv2 grid resolve
the BLOCK's pose -- especially orientation theta?

Regresses block (x, y, cos t, sin t) from the cached start+goal patch grids with
(a) exact linear ridge and (b) a small MLP (nonlinear ceiling). Reports x/y MAE
in px and theta MAE in degrees. If theta is poorly resolved, the oracle ceiling
(0.5) will cap below 0.90 -- this quantifies it BEFORE we blame `g`.

    cd dino_wm
    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python analysis/pose_probe.py \
        --data_path data/pusht_multicolor --split train
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.probe_common import load_probe_data, build_pose_dataset, episode_split, SIM


def pose_targets(pose):
    return torch.stack([pose[:, 0], pose[:, 1], torch.cos(pose[:, 2]), torch.sin(pose[:, 2])], dim=1)


def angular_mae_deg(theta_true, cos_p, sin_p):
    theta_pred = torch.atan2(sin_p, cos_p)
    d = torch.abs(theta_true - theta_pred) % (2 * np.pi)
    d = torch.minimum(d, 2 * np.pi - d)
    return float(torch.rad2deg(d).mean()), torch.rad2deg(d)


def ridge_dual(Xtr, Ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    K = Xtr @ Xtr.T
    A = K + lam * torch.eye(K.shape[0])
    alpha = torch.linalg.solve(A, Ytr)
    return (Xte @ Xtr.T) @ alpha


def train_mlp(Xtr, Ytr, Xte, epochs=300, lr=1e-3, wd=1e-4, device="cpu"):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = ((Xtr - mu) / sd).to(device), ((Xte - mu) / sd).to(device)
    # normalize x,y to ~[0,1] for stable training; cos/sin already O(1)
    scale = torch.tensor([SIM, SIM, 1.0, 1.0])
    Ytr_n = (Ytr / scale).to(device)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Linear(256, 4)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    bs = min(256, Xtr.shape[0])
    for _ in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(net(Xtr[idx]), Ytr_n[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        return (net(Xte).cpu() * scale)


def report_metrics(name, pred, pose_te):
    x_mae = float((pred[:, 0] - pose_te[:, 0]).abs().mean())
    y_mae = float((pred[:, 1] - pose_te[:, 1]).abs().mean())
    th_mae, th_err = angular_mae_deg(pose_te[:, 2], pred[:, 2], pred[:, 3])
    print(f"  {name:6s}: x_MAE={x_mae:6.1f}px  y_MAE={y_mae:6.1f}px  theta_MAE={th_mae:6.1f}deg  "
          f"theta_median={float(th_err.median()):.1f}deg")
    return {"x_mae_px": x_mae, "y_mae_px": y_mae, "theta_mae_deg": th_mae,
            "theta_median_deg": float(th_err.median())}, th_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_multicolor"))
    ap.add_argument("--latent_dir", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--max_samples", type=int, default=6000, help="cap for the ridge kernel")
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--out", default="analysis_outputs/pose_probe.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    latent_dir = args.latent_dir or os.path.join(args.data_path, "latents")

    data = load_probe_data(args.data_path, latent_dir, args.split)
    Xg, pose, ep = build_pose_dataset(data)
    X = Xg.reshape(Xg.shape[0], -1)  # (M, 196*384)
    n_eps = len(data["labels"])
    tr, te = episode_split(ep, n_eps, frac=args.test_frac)

    Xtr, Xte = X[tr], X[te]
    pose_tr, pose_te = pose[tr], pose[te]
    if Xtr.shape[0] > args.max_samples:
        keep = torch.randperm(Xtr.shape[0])[:args.max_samples]
        Xtr, pose_tr = Xtr[keep], pose_tr[keep]
    print(f"pose probe: train {Xtr.shape[0]} / test {Xte.shape[0]} samples, dim {X.shape[1]}")

    Ytr = pose_targets(pose_tr)
    out = {"split": args.split, "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0])}

    print("Block pose recovery (lower is better):")
    ridge_pred = ridge_dual(Xtr, Ytr, Xte, lam=args.ridge_lambda)
    out["ridge"], th_err_ridge = report_metrics("ridge", ridge_pred, pose_te)
    mlp_pred = train_mlp(Xtr, Ytr, Xte, device=args.device)
    out["mlp"], th_err_mlp = report_metrics("mlp", mlp_pred, pose_te)

    # success-threshold context: pose criterion uses theta < pi/9 = 20deg
    out["theta_tol_deg_pushT"] = 20.0
    out["frac_within_tol_mlp"] = float((th_err_mlp < 20.0).float().mean())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    # theta scatter
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, pred, name in [(ax[0], ridge_pred, "ridge"), (ax[1], mlp_pred, "mlp")]:
        tp = torch.atan2(pred[:, 3], pred[:, 2])
        a.scatter(np.rad2deg(pose_te[:, 2]), np.rad2deg(tp), s=8, alpha=0.5)
        a.plot([-180, 180], [-180, 180], "r--", lw=1)
        a.set_xlabel("theta true (deg)"); a.set_ylabel("theta pred (deg)"); a.set_title(name)
    fig.tight_layout()
    fig.savefig(args.out.replace(".json", "_theta.png"), dpi=110)
    print(f"\nReport -> {args.out} ; theta scatter -> {args.out.replace('.json', '_theta.png')}")


if __name__ == "__main__":
    main()
