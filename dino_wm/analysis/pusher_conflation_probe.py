"""Is the 'direction leak' just the PUSHER (circular), or the scene?

Objection (correct, if it holds): z_start includes the pusher, and the pusher sits
where it does BECAUSE the policy already committed to the goal direction ("pusher
left -> T moves right" is contact physics, not goal info). So a probe reading the
committed pusher off the start frame trivially predicts the natural continuation and
says NOTHING about whether language could specify direction.

Decisive cheap test at the STATE level (an upper bound on what any latent could
extract -- the latent is a function of the state): predict the block's displacement
direction (and rotation) from different feature subsets, held out by trajectory:
  block_only    : block pose [bx,by,cos,sin]        (the scene minus the pusher)
  pusher_rel    : pusher MINUS block [px-bx,py-by]  (contact geometry)
  pusher_abs    : pusher [px,py]
  full          : all
If pusher_rel predicts direction but block_only is ~chance -> the leak is the pusher
(circular), direction is NOT scene-determined, and language CAN specify it from a
pusher-neutral start. If block_only still predicts -> residual scene-determinism
(walls/position).

  DATASET_DIR=/workspace/data python analysis/pusher_conflation_probe.py
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.subsegment_extract import load_split, extract_candidates, dual_bound_mask, wrap_pi


def feats(data, traj, i):
    s = data["states5"][traj, i]                       # (M,5) [px,py,bx,by,th]
    px, py, bx, by, th = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
    block = np.stack([bx, by, np.cos(th), np.sin(th)], 1)
    p_rel = np.stack([px - bx, py - by], 1)
    p_abs = np.stack([px, py], 1)
    return {"block_only": block, "pusher_rel": p_rel, "pusher_abs": p_abs,
            "full": np.concatenate([block, p_rel], 1)}


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    """Closed-form ridge on standardized features+targets; held-out R2 (per-dim mean)."""
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-6
    my, sy = ytr.mean(0), ytr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mx) / sx, (Xte - mx) / sx
    Ytr = (ytr - my) / sy
    Xtr1 = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte1 = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    A = Xtr1.T @ Xtr1 + lam * np.eye(Xtr1.shape[1])
    W = np.linalg.solve(A, Xtr1.T @ Ytr)
    pred = Xte1 @ W
    yz = (yte - my) / sy
    ss_res = ((pred - yz) ** 2).sum(0); ss_tot = ((yz - yz.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss_res / ss_tot).mean())


def mlp_r2(Xtr, ytr, Xte, yte, device="cpu"):
    """Small MLP (captures the nonlinear pusher_rel->direction map) for an upper bound."""
    import torch, torch.nn as nn
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-6
    my, sy = ytr.mean(0), ytr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mx) / sx, dtype=torch.float32)
    Ytr_t = torch.tensor((ytr - my) / sy, dtype=torch.float32)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                      nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ytr.shape[1]))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.MSELoss()
    for _ in range(300):
        opt.zero_grad(); lossf(m(Xtr_t), Ytr_t).backward(); opt.step()
    with torch.no_grad():
        pred = m(torch.tensor((Xte - mx) / sx, dtype=torch.float32)).numpy()
    yz = (yte - my) / sy
    ss_res = ((pred - yz) ** 2).sum(0); ss_tot = ((yz - yz.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss_res / ss_tot).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATASET_DIR", ".") + "/pusht_noise")
    ap.add_argument("--out", default="analysis/out/pusher_conflation.json")
    args = ap.parse_args()
    data = load_split(args.data_path, "train")
    N = len(data["seqlen"]); tid = np.arange(N)
    test = tid % 5 == 0; train = ~test
    report = {}
    for h in [16, 48]:
        c = extract_candidates(data, h, 2)
        keep = dual_bound_mask(c, 1e9, 1e9, 15.0, 5.0)          # meaningful motion, cap off
        tr = keep & train[c["traj"]]; te = keep & test[c["traj"]]
        itr, ite = np.where(tr)[0], np.where(te)[0]
        rng = np.random.RandomState(0)
        if len(itr) > 8000: itr = rng.choice(itr, 8000, replace=False)
        if len(ite) > 4000: ite = rng.choice(ite, 4000, replace=False)
        Ftr = feats(data, c["traj"][itr], c["i"][itr])
        Fte = feats(data, c["traj"][ite], c["i"][ite])
        dir_tr, dir_te = c["dp"][itr], c["dp"][ite]
        rot_tr = np.degrees(c["drot"][itr])[:, None]; rot_te = np.degrees(c["drot"][ite])[:, None]
        print(f"\n=== h={h}  (n_test={len(ite)}; targets: direction dx/dy, rotation deg) ===")
        report[h] = {}
        for name in ["block_only", "pusher_rel", "pusher_abs", "full"]:
            d_lin = ridge_r2(Ftr[name], dir_tr, Fte[name], dir_te)
            d_mlp = mlp_r2(Ftr[name], dir_tr, Fte[name], dir_te)
            r_mlp = mlp_r2(Ftr[name], rot_tr, Fte[name], rot_te)
            report[h][name] = {"dir_R2_linear": round(d_lin, 3), "dir_R2_mlp": round(d_mlp, 3),
                               "rot_R2_mlp": round(r_mlp, 3), "dim": Ftr[name].shape[1]}
            print(f"  {name:11s} (d={Ftr[name].shape[1]})  direction R2: lin={d_lin:+.3f} mlp={d_mlp:+.3f}   "
                  f"rotation R2(mlp)={r_mlp:+.3f}")
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\n[out] -> {args.out}")
    print("\nINTERPRETATION: if pusher_rel >> block_only on direction -> the leak is the "
          "PUSHER (circular); direction is language-specifiable from a pusher-neutral start.")


if __name__ == "__main__":
    main()
