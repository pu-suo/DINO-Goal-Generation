"""Decisive G2 diagnostic: does the trained dynamics predict the MOVING block (I3 block-TF
latent error on CHANGED patches), vs copy-last? Aggregate patch-L2 is dominated by static
patches where copy-last is ~perfect; this isolates the moving region.

Run (dino_wm env): python lt_g2_blockcheck.py --cache /workspace/lt_cache --model /workspace/g2/model.pth
"""
import argparse
import numpy as np
import torch
from lt_g2 import Dyn, NP


def valid_stack(c, key):
    return np.concatenate([c[key][i, :c["seq_lengths"][i]] for i in range(len(c["seq_lengths"]))], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/lt_cache")
    ap.add_argument("--model", default="/workspace/g2/model.pth")
    ap.add_argument("--num_hist", type=int, default=3)
    ap.add_argument("--num_pred", type=int, default=1)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--topk", type=int, default=12)  # ~ patches a block+motion occupies
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    tr = dict(np.load(f"{args.cache}/train.npz", allow_pickle=True))
    va = dict(np.load(f"{args.cache}/val.npz", allow_pickle=True))
    fs = int(tr["frameskip"]); nh, npd = args.num_hist, args.num_pred; nf = nh + npd
    pm, ps = valid_stack(tr, "proprio").mean(0), valid_stack(tr, "proprio").std(0) + 1e-6
    am, as_ = valid_stack(tr, "actions").mean(0), valid_stack(tr, "actions").std(0) + 1e-6
    va["proprio_n"] = ((va["proprio"] - pm) / ps).astype(np.float32)
    va["actions_n"] = ((va["actions"] - am) / as_).astype(np.float32)

    model = Dyn(nh, npd, fs).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device)["model"])
    model.eval()

    seq = va["seq_lengths"]; valid = np.where(seq >= nf)[0]
    mc, cc, ma, ca, mot = [], [], [], [], []
    with torch.no_grad():
        for _ in range(args.n):
            i = valid[rng.randint(len(valid))]; s = rng.randint(0, seq[i] - nf + 1)
            v = torch.tensor(va["visual"][i, s:s + nf].astype(np.float32), device=device)[None]
            p = torch.tensor(va["proprio_n"][i, s:s + nf], device=device)[None]
            a = torch.tensor(va["actions_n"][i, s:s + nf], device=device)[None]
            z = model.assemble(v, p, a)
            z_pred = model.predict(z[:, :nh])
            pv = z_pred[0, -1, :NP]          # predicted frame nh (last step)
            tv = z[0, nh, :NP]               # true frame nh
            cl = z[0, nh - 1, :NP]           # copy-last (frame nh-1)
            motion = torch.linalg.norm(tv - cl, dim=-1)          # per-patch motion magnitude
            ch = motion.topk(args.topk).indices                  # the moving patches
            model_err = torch.linalg.norm(pv - tv, dim=-1)
            copy_err = torch.linalg.norm(cl - tv, dim=-1)
            mc.append(model_err[ch].mean().item()); cc.append(copy_err[ch].mean().item())
            ma.append(model_err.mean().item()); ca.append(copy_err.mean().item())
            mot.append(motion[ch].mean().item())
    mc, cc, ma, ca, mot = map(lambda x: float(np.mean(x)), (mc, cc, ma, ca, mot))
    print(f"n={args.n} windows, topk={args.topk} changed patches")
    print(f"  CHANGED patches:  model TF-L2={mc:.3f}   copy-last TF-L2={cc:.3f}   "
          f"(motion magnitude={mot:.3f})   model beats copy: {mc < cc}  (reduction {100*(1-mc/cc):.0f}%)")
    print(f"  ALL patches:      model TF-L2={ma:.3f}   copy-last TF-L2={ca:.3f}")
    print(f"  VERDICT: dynamics predicts block motion better than no-motion on the moving region: {mc < cc}")


if __name__ == "__main__":
    main()
