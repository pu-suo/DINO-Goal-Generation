"""Part D (TOP RISK, run EARLY): is the relative-language command load-bearing, or
does the static start frame already telegraph the goal via pusher->block contact
geometry?  This is the go/no-go for the whole relative-language direction. It runs
BEFORE the expensive Part-E retrain, on a GPU-encoded SAMPLE of A's sub-segments.

Faithful to what g actually sees: z_start = DinoV2 latent (196x384) of the CLEAN
rendered start frame (green-T removed, with_target=False) -- the SAME encode path
as cache_clean_dyn_latents / encode_obs. Held-out by TRAJECTORY (disjoint pools),
so a probe must GENERALIZE, not memorize the 1-goal-per-start mapping.

D1 (leak magnitude): can a probe predict the command (8-way direction / magnitude
   band / rotation sign) from z_start ALONE on held-out trajectories? Accuracy >>
   the majority-class prior == the leak. The contact-geom corr 0.65 (Part A) is the
   state-level upper bound; this asks whether the DINO LATENT preserves it.
D2 (the decider): predict the continuous block Dpose from
   A: z_start         B: z_start + command       C: z_start + SWAPPED command
   Language is load-bearing iff B beats A (command adds goal info beyond the frame)
   AND C degrades vs B (the prediction actually USES the command). Per the spec the
   ablation -- not the D1 correlation -- is the verdict.

If NOT load-bearing -> STOP: the pusher-decorrelation-vs-reachability fork (random-
izing the pusher start decorrelates it but breaks A3 replay-reachability). Surface,
do not paper over.

Box (4090):
  DATASET_DIR=/workspace/data /workspace/envs/dino_wm/bin/python \
    analysis/part_d_leak_probe.py --n_train 6000 --n_test 2000 \
    --cache analysis/out/part_d_cache.pt --out analysis/out/part_d.json
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.subsegment_extract import (
    load_split, extract_candidates, dual_bound_mask, bucketize, wrap_pi)
from datasets.img_transforms import default_transform
from datasets.rigid_goal_render import make_env, render_state
from models.dino import DinoV2Encoder


# --- sampling: disjoint-trajectory pools -------------------------------------
def sample_pool(data, traj_mask, dials, n, rng):
    """Sample n sub-segments whose trajectory is in traj_mask (bool over trajs)."""
    c = extract_candidates(data, dials["h"], dials["stride"])
    keep = dual_bound_mask(c, dials["D_max"], dials["R_max"], dials["D_min"], dials["R_min"])
    keep &= traj_mask[c["traj"]]
    idx = np.where(keep)[0]
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return {k: c[k][idx] for k in ("traj", "i", "j", "dp", "dp_mag", "drot")}


# --- render + encode (faithful to g's input) ---------------------------------
def encode_starts(sub, data, batch=128, device="cuda"):
    """Render CLEAN start frames for each sub-segment and DINO-encode -> (N,196,384)."""
    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
    enc_resize = transforms.Resize((224 // 16) * encoder.patch_size)
    tfm = default_transform(224)
    env = make_env(with_target=False)
    N = len(sub["i"])
    Z = torch.zeros(N, 196, encoder.emb_dim, dtype=torch.float16)
    buf, pos = [], []
    def flush():
        if not buf:
            return
        x = enc_resize(torch.stack(buf).to(device))
        with torch.no_grad():
            z = encoder.forward(x).cpu().half()
        for p, zz in zip(pos, z):
            Z[p] = zz
        buf.clear(); pos.clear()
    for k in range(N):
        t, i = int(sub["traj"][k]), int(sub["i"][k])
        s5 = data["states5"][t, i]
        img, _ = render_state(env, s5)                          # clean 224x224x3 uint8
        x = torch.tensor(img).float().div_(255.0).permute(2, 0, 1)
        buf.append(tfm(x)); pos.append(k)
        if len(buf) >= batch:
            flush()
        if (k + 1) % 1000 == 0:
            print(f"  encoded {k+1}/{N}", flush=True)
    flush()
    return Z


def labels_from_sub(sub):
    """Command buckets + continuous Dpose target from sub-segment geometry."""
    dbin, mbin, rsign, rmbin = bucketize(sub["dp"], sub["drot"])
    dpose = np.stack([sub["dp"][:, 0], sub["dp"][:, 1], np.degrees(sub["drot"])], axis=1)
    cmd_onehot = np.concatenate([
        np.eye(8)[dbin], np.eye(3)[np.clip(mbin, 0, 2)],
        np.eye(3)[rsign + 1], np.eye(3)[np.clip(rmbin, 0, 2)]], axis=1)  # (N,17)
    return dict(dir=dbin, mag=mbin, rsign=rsign, rmag=rmbin,
                dpose=dpose.astype(np.float32), cmd=cmd_onehot.astype(np.float32))


# --- probes ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_hid=512, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_out))
    def forward(self, x):
        return self.net(x)


def train_probe(Xtr, ytr, Xte, yte, task, device, epochs=120, wd=1e-3, lr=1e-3):
    """task='cls' -> accuracy; task='reg' -> per-dim R2 (held-out)."""
    d_in = Xtr.shape[1]
    if task == "cls":
        d_out = int(max(ytr.max(), yte.max())) + 1
        model = MLP(d_in, d_out).to(device)
        lossf = nn.CrossEntropyLoss()
        Ytr = torch.as_tensor(ytr, dtype=torch.long)
    else:
        d_out = ytr.shape[1]
        model = MLP(d_in, d_out).to(device)
        lossf = nn.MSELoss()
        Ytr = torch.as_tensor(ytr, dtype=torch.float32)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr_t); bs = 256
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            xb = Xtr_t[b].to(device); yb = Ytr[b].to(device)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        ptr = model(Xtr_t.to(device)).cpu()
        pte = model(torch.as_tensor(Xte, dtype=torch.float32).to(device)).cpu()
    if task == "cls":
        acc_tr = float((ptr.argmax(1).numpy() == ytr).mean())
        acc_te = float((pte.argmax(1).numpy() == yte).mean())
        prior = float(np.bincount(ytr, minlength=ptr.shape[1]).max() / len(ytr))
        return dict(train_acc=round(acc_tr, 4), test_acc=round(acc_te, 4),
                    prior=round(prior, 4), n_classes=int(ptr.shape[1]))
    else:
        yte = np.asarray(yte, dtype=np.float32)
        mse = ((pte.numpy() - yte) ** 2).mean(0)
        var = yte.var(0) + 1e-8
        r2 = 1 - mse / var
        return dict(r2_per_dim=[round(float(v), 4) for v in r2],
                    r2_mean=round(float(r2.mean()), 4),
                    mse_per_dim=[round(float(v), 3) for v in mse])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATASET_DIR", ".") + "/pusht_noise")
    ap.add_argument("--split", default="train")
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--D_max", type=float, default=50.0); ap.add_argument("--R_max", type=float, default=12.0)
    ap.add_argument("--D_min", type=float, default=15.0); ap.add_argument("--R_min", type=float, default=5.0)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--test_mod", type=int, default=5, help="traj_id %% test_mod == 0 -> held-out pool")
    ap.add_argument("--cache", default=None, help="cache encoded latents+labels here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    dials = dict(h=args.h, stride=args.stride, D_max=args.D_max, R_max=args.R_max,
                 D_min=args.D_min, R_min=args.R_min)

    if args.cache and Path(args.cache).exists():
        print(f"[cache] loading {args.cache}")
        blob = torch.load(args.cache)
        Ztr, Zte, Ltr, Lte = blob["Ztr"], blob["Zte"], blob["Ltr"], blob["Lte"]
    else:
        data = load_split(args.data_path, args.split)
        N = len(data["seqlen"])
        tid = np.arange(N)
        test_mask = (tid % args.test_mod == 0)
        train_mask = ~test_mask
        assert not (train_mask & test_mask).any(), "traj pools overlap"
        print(f"[split] {train_mask.sum()} train-pool trajs / {test_mask.sum()} held-out trajs (disjoint)")
        sub_tr = sample_pool(data, train_mask, dials, args.n_train, rng)
        sub_te = sample_pool(data, test_mask, dials, args.n_test, rng)
        assert len(np.intersect1d(np.unique(sub_tr["traj"]), np.unique(sub_te["traj"]))) == 0, \
            "LEAK: shared trajectory across train/test probe pools"
        print(f"[sample] {len(sub_tr['i'])} train / {len(sub_te['i'])} held-out sub-segments")
        print("[encode] train start frames...");  Ztr = encode_starts(sub_tr, data, device=device)
        print("[encode] held-out start frames..."); Zte = encode_starts(sub_te, data, device=device)
        Ltr, Lte = labels_from_sub(sub_tr), labels_from_sub(sub_te)
        if args.cache:
            Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(Ztr=Ztr, Zte=Zte, Ltr=Ltr, Lte=Lte, dials=dials), args.cache)
            print(f"[cache] -> {args.cache}")

    Xtr = Ztr.reshape(len(Ztr), -1).float().numpy()
    Xte = Zte.reshape(len(Zte), -1).float().numpy()
    print(f"[probe] z_start flattened dim = {Xtr.shape[1]}")
    report = {"dials": dials, "n_train": len(Xtr), "n_test": len(Xte)}

    # --- D1: can z_start ALONE predict the command? (held-out) ---------------
    print("\n=== D1: leak -- predict command from z_start alone (held-out) ===")
    report["D1"] = {}
    for name, key in (("direction(8)", "dir"), ("magnitude", "mag"), ("rot_sign", "rsign")):
        y_tr = Ltr[key].copy(); y_te = Lte[key].copy()
        if key == "rsign":          # map -1,0,1 -> 0,1,2
            y_tr = y_tr + 1; y_te = y_te + 1
        r = train_probe(Xtr, y_tr, Xte, y_te, "cls", device)
        report["D1"][name] = r
        lift = r["test_acc"] - r["prior"]
        print(f"  {name:13s}  test_acc={r['test_acc']:.3f}  prior={r['prior']:.3f}  "
              f"lift={lift:+.3f}  (train_acc={r['train_acc']:.3f})")

    # --- D2: does language add goal info beyond z_start? (Dpose regression) ---
    print("\n=== D2: ablation -- predict continuous Dpose (held-out R2) ===")
    # standardize targets by TRAIN stats so the MSE loss balances (dx,dy,drot) and the
    # probe does not underfit rotation (R2 is affine-invariant -> same as raw R2).
    mu = Ltr["dpose"].mean(0); sd = Ltr["dpose"].std(0) + 1e-6
    ytr_d = (Ltr["dpose"] - mu) / sd
    yte_d = (Lte["dpose"] - mu) / sd
    cmd_tr, cmd_te = Ltr["cmd"], Lte["cmd"]
    swap = rng.permutation(len(cmd_te))                          # swapped-language at eval
    A_tr = Xtr; A_te = Xte
    B_tr = np.concatenate([Xtr, cmd_tr], axis=1)
    B_te = np.concatenate([Xte, cmd_te], axis=1)
    C_te = np.concatenate([Xte, cmd_te[swap]], axis=1)           # wrong command
    rA = train_probe(A_tr, ytr_d, A_te, yte_d, "reg", device)
    # train B once; evaluate on correct (B) and swapped (C) commands
    modelB_eval = _train_reg_two_eval(B_tr, ytr_d, B_te, C_te, yte_d, device)
    report["D2"] = {"A_start_only": rA, "B_start_plus_lang": modelB_eval["B"],
                    "C_swapped_lang": modelB_eval["C"]}
    print(f"  A start-only        R2_mean={rA['r2_mean']:.3f}  per-dim(dx,dy,drot)={rA['r2_per_dim']}")
    print(f"  B start+language    R2_mean={modelB_eval['B']['r2_mean']:.3f}  per-dim={modelB_eval['B']['r2_per_dim']}")
    print(f"  C start+SWAPPED     R2_mean={modelB_eval['C']['r2_mean']:.3f}  per-dim={modelB_eval['C']['r2_per_dim']}")

    # --- verdict --------------------------------------------------------------
    dlift = report["D1"]["direction(8)"]["test_acc"] - report["D1"]["direction(8)"]["prior"]
    b_minus_a = modelB_eval["B"]["r2_mean"] - rA["r2_mean"]
    b_minus_c = modelB_eval["B"]["r2_mean"] - modelB_eval["C"]["r2_mean"]
    load_bearing = (b_minus_a > 0.05) and (b_minus_c > 0.05)
    report["verdict"] = {"direction_leak_lift": round(float(dlift), 4),
                         "B_minus_A_r2": round(float(b_minus_a), 4),
                         "B_minus_C_r2": round(float(b_minus_c), 4),
                         "language_load_bearing": bool(load_bearing)}
    print("\n=== VERDICT ===")
    print(f"  direction leak lift (test_acc - prior) = {dlift:+.3f}")
    print(f"  language adds goal info (R2 B-A) = {b_minus_a:+.3f}")
    print(f"  prediction uses language (R2 B-C) = {b_minus_c:+.3f}")
    print(f"  -> language LOAD-BEARING: {load_bearing}")
    if not load_bearing:
        print("  STOP: pusher leak dominates. Fork = pusher-decorrelation vs replay-reachability.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2, default=str)
        print(f"\n[out] -> {args.out}")
    return report


def _train_reg_two_eval(Xtr, ytr, Xte_B, Xte_C, yte, device, epochs=120, wd=1e-3, lr=1e-3):
    """Train one regression probe, evaluate on two different test inputs (B correct,
    C swapped-language) against the same targets."""
    model = MLP(Xtr.shape[1], ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32); Yt = torch.as_tensor(ytr, dtype=torch.float32)
    lossf = nn.MSELoss(); n = len(Xt); bs = 256
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad(); lossf(model(Xt[b].to(device)), Yt[b].to(device)).backward(); opt.step()
    model.eval(); out = {}
    yte = np.asarray(yte, dtype=np.float32); var = yte.var(0) + 1e-8
    with torch.no_grad():
        for name, X in (("B", Xte_B), ("C", Xte_C)):
            p = model(torch.as_tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
            mse = ((p - yte) ** 2).mean(0); r2 = 1 - mse / var
            out[name] = dict(r2_per_dim=[round(float(v), 4) for v in r2],
                             r2_mean=round(float(r2.mean()), 4))
    return out


if __name__ == "__main__":
    main()
