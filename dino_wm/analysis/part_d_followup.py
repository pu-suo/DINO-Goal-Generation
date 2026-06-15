"""Part D follow-up: drowning-free per-component ablation on the CACHED latents.

The main D2 ablation appends a 17-d command to the 75264-d z_start, so the command
can be drowned and the mean-R2 verdict can under-credit non-leaked components. Here
we use RESIDUAL boosting, isolating each command component:
  1. fit A: z_start -> target  (held-out R2_A = what the start frame leaks)
  2. residual r = target - A(z_start)
  3. fit R: command_subset -> r  (command is the ONLY input -> not drowned)
  4. compare R2 of  A,  A+R(correct cmd),  A+R(SWAPPED cmd), and R(cmd) vs residual.
A component is LOAD-BEARING iff its command predicts the residual (R2_resid>0) AND the
correct command beats the swapped one. Runs on analysis/out/part_d_cache.pt (no encode).
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_hid=512, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(d_hid, d_out))
    def forward(self, x): return self.net(x)


def fit_reg(Xtr, ytr, Xte, device, epochs=150, wd=1e-3, lr=1e-3, hid=512):
    Xt = torch.as_tensor(Xtr, dtype=torch.float32); Yt = torch.as_tensor(ytr, dtype=torch.float32)
    if Yt.ndim == 1: Yt = Yt[:, None]
    m = MLP(Xtr.shape[1], Yt.shape[1], d_hid=hid).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd); lossf = nn.MSELoss()
    n = len(Xt); bs = 256
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n)
        for s in range(0, n, bs):
            b = perm[s:s+bs]; opt.zero_grad()
            lossf(m(Xt[b].to(device)), Yt[b].to(device)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.as_tensor(Xte, dtype=torch.float32).to(device)).cpu().numpy()


def r2(y, p):
    y = np.asarray(y, dtype=np.float32); p = np.asarray(p, dtype=np.float32).reshape(y.shape)
    return float(1 - ((p - y) ** 2).mean() / (y.var() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="analysis/out/part_d_cache.pt")
    ap.add_argument("--out", default="analysis/out/part_d_followup.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    blob = torch.load(args.cache)
    Xtr = blob["Ztr"].reshape(len(blob["Ztr"]), -1).float().numpy()
    Xte = blob["Zte"].reshape(len(blob["Zte"]), -1).float().numpy()
    Ltr, Lte = blob["Ltr"], blob["Lte"]
    swap = rng.permutation(len(Xte))

    # command subsets (one-hot layout: dir8 | mag3 | rsign3 | rmag3 = 17)
    cmd_tr, cmd_te = Ltr["cmd"], Lte["cmd"]
    subsets = {"full": slice(0, 17), "dir": slice(0, 8), "mag": slice(8, 11),
               "rot": slice(11, 17)}
    # targets: displacement components + rotation (deg)
    tgt = {"dx": (Ltr["dpose"][:, 0], Lte["dpose"][:, 0]),
           "dy": (Ltr["dpose"][:, 1], Lte["dpose"][:, 1]),
           "drot": (Ltr["dpose"][:, 2], Lte["dpose"][:, 2])}

    report = {}
    print(f"[followup] n_train={len(Xtr)} n_test={len(Xte)} (cached latents, no re-encode)")
    for tname, (ytr, yte) in tgt.items():
        # standardize target for stable training (R2 is scale-invariant)
        mu, sd = ytr.mean(), ytr.std() + 1e-6
        ytr_s, yte_s = (ytr - mu) / sd, (yte - mu) / sd
        # A: z_start -> target  (what the start frame leaks)
        predA_te = fit_reg(Xtr, ytr_s, Xte, device)
        r2A = r2(yte_s, predA_te)
        predA_tr = fit_reg(Xtr, ytr_s, Xtr, device)         # for residual targets
        resid_tr = ytr_s - predA_tr.ravel()
        resid_te = yte_s - predA_te.ravel()
        row = {"R2_start_only(A)": round(r2A, 4)}
        for sname, sl in subsets.items():
            predR_te = fit_reg(cmd_tr[:, sl], resid_tr, cmd_te[:, sl], device, hid=64)
            # how much of the A-residual does THIS command subset explain (held-out)?
            r2_resid = r2(resid_te, predR_te)
            # full prediction A + R(correct) vs A + R(swapped command)
            predR_sw = predR_te[swap]
            r2_AB = r2(yte_s, predA_te.ravel() + predR_te.ravel())
            r2_AC = r2(yte_s, predA_te.ravel() + predR_sw.ravel())
            row[sname] = {"R2_resid": round(r2_resid, 4),
                          "R2_A+R": round(r2_AB, 4), "R2_A+swap": round(r2_AC, 4),
                          "uses_lang": bool(r2_resid > 0.03 and (r2_AB - r2_AC) > 0.03)}
        report[tname] = row
        print(f"\n[{tname}]  A(start-only) R2={r2A:.3f}")
        for sname in subsets:
            d = row[sname]
            print(f"   cmd[{sname:4s}] -> residual R2={d['R2_resid']:+.3f}   "
                  f"A+R={d['R2_A+R']:.3f}  A+swap={d['R2_A+swap']:.3f}  "
                  f"uses_lang={d['uses_lang']}")

    # headline: is rotation (the non-leaked axis) load-bearing via the rot command?
    rot_lb = report["drot"]["rot"]["uses_lang"]
    dx_dir_lb = report["dx"]["dir"]["uses_lang"] or report["dy"]["dir"]["uses_lang"]
    report["headline"] = {
        "direction_leaked_into_start": True,   # from D1 lift +0.49 (reported separately)
        "rotation_command_load_bearing": bool(rot_lb),
        "displacement_dir_command_adds_over_start": bool(dx_dir_lb),
    }
    print("\n=== FOLLOW-UP HEADLINE ===")
    print(f"  rotation command load-bearing (on drot residual): {rot_lb}")
    print(f"  direction command adds over start (dx/dy):        {dx_dir_lb}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print(f"[out] -> {args.out}")


if __name__ == "__main__":
    main()
