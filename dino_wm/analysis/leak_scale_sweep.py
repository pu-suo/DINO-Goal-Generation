"""Leak-vs-scale sweep (Part 0 anchor + Part 2). Tests whether the Part-D direction
leak (z_start -> goal direction, R2~0.72 at h=16) is a SHORT-HORIZON ARTIFACT: does
direction un-leak as goals grow, while rotation stays un-leaked?

g-FREE. Measures task structure only. Held out by TRAJECTORY. Faithful z_start
(same DinoV2 encode as Part D). Varies ONLY h (window); the meaningful-motion LOWER
bound is held fixed (D_min=15 OR R_min=5) so a 'direction' exists to predict, while
the D_max/R_max UPPER caps are REMOVED so scale is free (spec Part 1).

Part 0 anchor reuses the Part-D cache (analysis/out/part_d_cache.pt, h=16 dual-bound)
to reproduce ~0.72 / ~0.47 and a shuffle control, confirming no drift from Part D.

  DATASET_DIR=/workspace/data python analysis/leak_scale_sweep.py \
    --cache analysis/out/part_d_cache.pt --out analysis/out/leak_scale.json
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.subsegment_extract import load_split, extract_candidates, wrap_pi
from analysis.part_d_followup import fit_reg, r2
from datasets.img_transforms import default_transform
from datasets.rigid_goal_render import make_env, render_state
from models.dino import DinoV2Encoder

CELL_PX = 100.0  # 3x3 grid over [100,400] -> 100px/cell


def probe_r2(Xtr, ytr, Xte, yte, device):
    """Standardize targets (R2 affine-invariant), fit MLP, return held-out R2
    (combined + per-component) and a shuffle-control R2."""
    ytr = np.atleast_2d(ytr.T).T if ytr.ndim == 1 else ytr
    yte = np.atleast_2d(yte.T).T if yte.ndim == 1 else yte
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-6
    pred = fit_reg(Xtr, (ytr - mu) / sd, Xte, device)
    yz = (yte - mu) / sd
    per = [round(r2(yz[:, d], pred[:, d]), 4) for d in range(yz.shape[1])]
    comb = round(float(1 - ((pred - yz) ** 2).sum() / ((yz - yz.mean(0)) ** 2).sum()), 4)
    # shuffle control: predictions vs trajectory-shuffled test targets -> ~0
    rng = np.random.RandomState(0)
    sh = rng.permutation(len(yz))
    shuf = round(float(1 - ((pred - yz[sh]) ** 2).sum() / ((yz - yz.mean(0)) ** 2).sum()), 4)
    return {"r2": comb, "r2_per_dim": per, "r2_shuffle_ctrl": shuf, "n_test": int(len(yz))}


def extract_at_scale(data, h, stride, traj_mask):
    """(frame_i, frame_{i+h}) for int h, or (frame_i, frame_{L-1}) for 'full', over
    trajectories in traj_mask. Returns dict of arrays + dp(2)/dp_mag/drot_deg."""
    if h == "full":
        T, I, J = [], [], []
        for t in range(len(data["seqlen"])):
            if not traj_mask[t]:
                continue
            L = int(data["seqlen"][t])
            if L < 20:
                continue
            s = np.arange(0, L - 1, stride)
            T.append(np.full(len(s), t)); I.append(s); J.append(np.full(len(s), L - 1))
        traj, i, j = np.concatenate(T), np.concatenate(I), np.concatenate(J)
    else:
        c = extract_candidates(data, int(h), stride)
        m = traj_mask[c["traj"]]
        traj, i, j = c["traj"][m], c["i"][m], c["j"][m]
    b0 = data["states5"][traj, i, 2:4]; b1 = data["states5"][traj, j, 2:4]
    dp = b1 - b0
    drot = np.degrees(wrap_pi(data["states5"][traj, j, 4] - data["states5"][traj, i, 4]))
    return dict(traj=traj, i=i, j=j, dp=dp, dp_mag=np.linalg.norm(dp, axis=1), drot=drot)


def lower_bound(cand, D_min=15.0, R_min=5.0):
    """Meaningful-motion lower bound (so a direction exists); NO upper cap."""
    return (cand["dp_mag"] >= D_min) | (np.abs(cand["drot"]) >= R_min)


def sample(cand, mask, n, rng):
    idx = np.where(mask)[0]
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return {k: cand[k][idx] for k in cand}


def build_encoder(device):
    enc = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
    return enc, transforms.Resize((224 // 16) * enc.patch_size), default_transform(224), make_env(with_target=False)


def encode_z(sub, data, enc, enc_resize, tfm, env, device, batch=128):
    N = len(sub["i"]); Z = torch.zeros(N, 196, enc.emb_dim, dtype=torch.float16)
    buf, pos = [], []
    def flush():
        if not buf: return
        x = enc_resize(torch.stack(buf).to(device))
        with torch.no_grad():
            z = enc.forward(x).cpu().half()
        for p, zz in zip(pos, z): Z[p] = zz
        buf.clear(); pos.clear()
    for k in range(N):
        img, _ = render_state(env, data["states5"][int(sub["traj"][k]), int(sub["i"][k])])
        x = torch.tensor(img).float().div_(255.0).permute(2, 0, 1)
        buf.append(tfm(x)); pos.append(k)
        if len(buf) >= batch: flush()
    flush()
    return Z.reshape(N, -1).float().numpy()


def dist(x, ps=(50, 90)):
    return {f"p{p}": round(float(np.percentile(x, p)), 1) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATASET_DIR", ".") + "/pusht_noise")
    ap.add_argument("--cache", default="analysis/out/part_d_cache.pt")
    ap.add_argument("--n_train", type=int, default=2500)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--test_mod", type=int, default=5)
    ap.add_argument("--out", default="analysis/out/leak_scale.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    report = {}

    # ---- Part 0: anchor to Part D (cached h=16 dual-bound) -------------------
    print("=== Part 0: anchor to Part D (cache, h=16 dual-bound) ===")
    blob = torch.load(args.cache)
    Xtr = blob["Ztr"].reshape(len(blob["Ztr"]), -1).float().numpy()
    Xte = blob["Zte"].reshape(len(blob["Zte"]), -1).float().numpy()
    dptr, dpte = blob["Ltr"]["dpose"], blob["Lte"]["dpose"]
    a_dir = probe_r2(Xtr, dptr[:, :2], Xte, dpte[:, :2], device)
    a_rot = probe_r2(Xtr, dptr[:, 2], Xte, dpte[:, 2], device)
    report["anchor_h16_dualbound"] = {"direction": a_dir, "rotation": a_rot}
    print(f"  direction(dx,dy) R2={a_dir['r2']:.3f} per={a_dir['r2_per_dim']} shuffle={a_dir['r2_shuffle_ctrl']:.3f}")
    print(f"  rotation(drot)   R2={a_rot['r2']:.3f} shuffle={a_rot['r2_shuffle_ctrl']:.3f}")
    ok = (0.60 <= a_dir["r2"] <= 0.85) and (abs(a_dir["r2_shuffle_ctrl"]) < 0.05)
    print(f"  ANCHOR {'OK (matches Part D ~0.72 + shuffle~0)' if ok else 'DRIFT -- reconcile before trusting sweep'}")
    report["anchor_ok"] = bool(ok)

    # ---- Part 2: leak vs scale ----------------------------------------------
    print("\n=== Part 2: leak vs scale (cap OFF; lower bound on) ===")
    data = load_split(args.data_path, "train")
    N = len(data["seqlen"]); tid = np.arange(N)
    test_mask = tid % args.test_mod == 0; train_mask = ~test_mask
    enc, enc_resize, tfm, env = build_encoder(device)
    report["sweep"] = {}
    for h in [16, 32, 48, 64, "full"]:
        ctr = extract_at_scale(data, h, args.stride, train_mask)
        cte = extract_at_scale(data, h, args.stride, test_mask)
        str_ = sample(ctr, lower_bound(ctr), args.n_train, rng)
        ste = sample(cte, lower_bound(cte), args.n_test, rng)
        Ztr = encode_z(str_, data, enc, enc_resize, tfm, env, device)
        Zte = encode_z(ste, data, enc, enc_resize, tfm, env, device)
        d_dir = probe_r2(Ztr, str_["dp"], Zte, ste["dp"], device)
        d_rot = probe_r2(Ztr, str_["drot"], Zte, ste["drot"], device)
        dmag = np.concatenate([str_["dp_mag"], ste["dp_mag"]])
        drot = np.abs(np.concatenate([str_["drot"], ste["drot"]]))
        report["sweep"][str(h)] = {
            "n_train": len(str_["i"]), "n_test": len(ste["i"]),
            "direction_R2": d_dir, "rotation_R2": d_rot,
            "disp_px": dist(dmag), "disp_cells": {k: round(v / CELL_PX, 2) for k, v in dist(dmag).items()},
            "rot_deg": dist(drot)}
        print(f"  h={str(h):4s} n={len(ste['i']):4d} | dir R2={d_dir['r2']:.3f}{d_dir['r2_per_dim']} "
              f"rot R2={d_rot['r2']:.3f} | disp p50/p90={dist(dmag)['p50']:.0f}/{dist(dmag)['p90']:.0f}px "
              f"({dist(dmag)['p50']/CELL_PX:.1f}/{dist(dmag)['p90']/CELL_PX:.1f} cells) "
              f"rot p50/p90={dist(drot)['p50']:.0f}/{dist(drot)['p90']:.0f}°")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print(f"\n[out] -> {args.out}")
    # headline
    s = report["sweep"]
    print("\n=== LEAK-VS-SCALE SUMMARY ===")
    print("  h     dir_R2  rot_R2  disp(cells p50/p90)")
    for h in ["16", "32", "48", "64", "full"]:
        r = s[h]
        print(f"  {h:5s} {r['direction_R2']['r2']:.3f}   {r['rotation_R2']['r2']:.3f}   "
              f"{r['disp_cells']['p50']}/{r['disp_cells']['p90']}")


if __name__ == "__main__":
    main()
