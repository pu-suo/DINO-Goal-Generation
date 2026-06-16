"""Rollout (multi-step latent-consistency) fine-tune -- fixes the exposure bias.

The base model is trained 1-step teacher-forced (lt_g2.tf_loss, num_pred=1): it only ever predicts
from GROUND-TRUTH latents, so at test time it goes OOD on its own predictions and the reliable
horizon collapses to ~1-2 steps (lt_d2_horizon: K=1=K=2=0.042u, drift from K=4). This fine-tunes
with an H-step ROLLOUT loss: unroll the model H steps on its OWN visual predictions (GT proprio +
GT action), MSE vs the GT latent at each step, backprop THROUGH the rollout. = DINO-WM's H-step
latent-consistency loss. Same data, same model, same encoder -- only the loss changes.

Window/predict alignment matches lt_g2.rollout_batch exactly: to predict frame t, feed the last nh
frames (visual=own predictions after the GT history; proprio/action = GT cache[t-nh:t]); take
predict(window)[:, -1, :NP].

Run: python lt_g2_rollout.py --cache /workspace/lt_cache_3k --init /workspace/g2_3k/model.pth \
     --out /workspace/g2_3k_roll --horizon 8 --epochs 8
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/dino_goal/dino_wm")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt_g2 import Dyn, NP, sample_windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/lt_cache_3k")
    ap.add_argument("--init", required=True)              # base 1-step model to fine-tune from
    ap.add_argument("--out", default="/workspace/g2_3k_roll")
    ap.add_argument("--num_hist", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=8)     # H rollout steps (target reliable horizon)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--iters_per_epoch", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    tr = dict(np.load(f"{a.cache}/train.npz", allow_pickle=True))
    va = dict(np.load(f"{a.cache}/val.npz", allow_pickle=True))
    fs = int(tr["frameskip"]); nh = a.num_hist; H = a.horizon; nf = nh + H

    def vstack(c, k):
        return np.concatenate([c[k][i, :int(c["seq_lengths"][i])] for i in range(len(c["seq_lengths"]))], 0).reshape(-1, c[k].shape[-1])
    pm, ps = vstack(tr, "proprio").mean(0), vstack(tr, "proprio").std(0) + 1e-6
    am, as_ = vstack(tr, "actions").mean(0), vstack(tr, "actions").std(0) + 1e-6
    for c in (tr, va):
        c["proprio_n"] = ((c["proprio"] - pm) / ps).astype(np.float32)
        c["actions_n"] = ((c["actions"] - am) / as_).astype(np.float32)

    m = Dyn(nh, 1, fs).to(dev)
    m.load_state_dict(torch.load(a.init, map_location=dev)["model"])
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    print(f"fine-tune from {a.init}  H={H}  nf={nf}  batch={a.batch}")

    def rollout_loss(vis, prop_n, act_n, per_step=False):
        # vis (b,nf,196,d), prop_n/act_n (b,nf,*). Autoregress visual; GT proprio+action.
        vlist = [vis[:, j] for j in range(nh)]
        losses = []
        for h in range(H):
            t = nh + h
            win_v = torch.stack(vlist[-nh:], 1)
            win_p = prop_n[:, t - nh:t]
            win_a = act_n[:, t - nh:t]
            pred = m.predict(m.assemble(win_v, win_p, win_a))[:, -1, :NP]
            losses.append(((pred - vis[:, t]) ** 2).mean())
            vlist.append(pred)
        if per_step:
            return torch.stack(losses)
        return torch.stack(losses).mean()

    def batch(c):
        v, p, act = sample_windows(c, nf, a.batch, rng)
        return (torch.tensor(v, device=dev), torch.tensor(p, device=dev), torch.tensor(act, device=dev))

    for e in range(1, a.epochs + 1):
        m.train(); tot = 0.0
        for _ in range(a.iters_per_epoch):
            loss = rollout_loss(*batch(tr))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        m.eval()
        with torch.no_grad():
            ps_ = rollout_loss(*batch(va), per_step=True).cpu().numpy()
        print(f"  epoch {e}: train roll-loss={tot/a.iters_per_epoch:.4f}  val roll-loss={ps_.mean():.4f}  "
              f"(step1={ps_[0]:.3f} stepH={ps_[-1]:.3f})")
        torch.save({"model": m.state_dict()}, os.path.join(a.out, "model.pth"))
    print(f"DONE -> {a.out}/model.pth  (H={H})")


if __name__ == "__main__":
    main()
