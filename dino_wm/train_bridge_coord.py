"""Train the COORDINATE bridge `g` on cached clean-scene latents (clean-scene pivot, Option A).

Isolated from the dynamics train.py (frozen-everything-else). Data = PushTCoordLatentGoalDataset:
(z_start, z_goal, spec=(x,y,theta) sim-512) from scripts/cache_coord_latents.py over the single-T
clean scene (scripts/gen_pusht_coord.py). The spec is decorrelated from the start pose, so `g`
must read it. Loss = weighted-L2 to enc(o_goal), up-weighted on changed patches (Otsu tau). Logs
the Stage-1 fidelity metric (changed-region cosine vs enc(o_goal)); the >=0.90 gate is checked
before wiring g into CEM, plus an identity floor and a SWAPPED-SPEC grounding check (the spec must
be load-bearing: a wrong spec should collapse changed-cos toward the identity floor).

Box (4090):
  /workspace/envs/dino_wm/bin/python train_bridge_coord.py \
    --latent_dir /workspace/data/pusht_coord/latents --out outputs/bridge/g_coord --epochs 100
Local smoke (cpu, tiny):
  /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python train_bridge_coord.py \
    --latent_dir data/pusht_coord_smoke/latents --out /tmp/g_coord_smoke --epochs 40 --batch_size 8
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.pusht_coord_dset import PushTCoordLatentGoalDataset
from models.bridge import (BridgeG, bridge_loss, changed_region_mask, estimate_tau,
                           DIM, N_PATCHES)


def collate(items):
    return {
        "z_start": torch.stack([it["z_start"] for it in items]),
        "z_goal": torch.stack([it["z_goal"] for it in items]),
        "spec": torch.stack([it["spec"] for it in items]),
    }


def changed_region_cosine(pred, target, z_start, tau):
    """Mean cosine(pred, target) over CHANGED patches -- the Stage-1 fidelity metric (§8)."""
    changed = changed_region_mask(z_start, target, tau)            # (B,196)
    cos = F.cosine_similarity(pred, target, dim=-1)                # (B,196)
    return (cos * changed).sum() / changed.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(g, loader, tau, lam, device, swap_spec=False):
    g.eval()
    tot, cosw, n = 0.0, 0.0, 0
    for b in loader:
        zs, zt, spec = b["z_start"].to(device), b["z_goal"].to(device), b["spec"].to(device)
        if swap_spec:                       # roll the spec so it no longer matches z_goal
            spec = torch.roll(spec, shifts=1, dims=0)
        zp = g.forward_coord(zs, spec)
        bs = zs.shape[0]
        tot += bridge_loss(zp, zt, zs, tau, lam, reduction="none").sum().item()
        cosw += float(changed_region_cosine(zp, zt, zs, tau)) * bs
        n += bs
    g.train()
    return tot / max(n, 1), cosw / max(n, 1)


@torch.no_grad()
def identity_floor(loader, tau, device):
    """changed-cos if g did NOTHING (pred=z_start): the 'copy-start' floor the spec must beat."""
    cosw, n = 0.0, 0
    for b in loader:
        zs, zt = b["z_start"].to(device), b["z_goal"].to(device)
        bs = zs.shape[0]
        cosw += float(changed_region_cosine(zs, zt, zs, tau)) * bs
        n += bs
    return cosw / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", required=True, help="<latent_dir>/<split>/{start,goal}_latents.pth + specs.pth")
    ap.add_argument("--out", default="outputs/bridge/g_coord")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--width", type=int, default=DIM)
    ap.add_argument("--n_freq", type=int, default=12, help="Fourier frequencies for (x,y)")
    ap.add_argument("--heat_sigma", type=float, default=1.2, help="Gaussian heatmap sigma (patch units)")
    ap.add_argument("--lam", type=float, default=7.0, help="changed-patch up-weight (§4 lambda)")
    ap.add_argument("--tau", type=float, default=None, help="override; default = Otsu on train set")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--fast_tf32", action="store_true", help="enable TF32 matmul (4090 speedup)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.fast_tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    train_dset = PushTCoordLatentGoalDataset(args.latent_dir, split="train")
    val_dset = PushTCoordLatentGoalDataset(args.latent_dir, split="val")
    print(f"data: train={len(train_dset)} val={len(val_dset)} | device={device}")

    tau = args.tau if args.tau is not None else estimate_tau(train_dset.start, train_dset.goal)
    frac = changed_region_mask(train_dset.start, train_dset.goal, tau).mean().item()
    print(f"[tau] tau={tau:.3f} -> {frac*100:.1f}% of patches flagged changed (train)")

    g = BridgeG(dim=args.width, depth=args.depth, heads=args.heads, cond_mode="coord",
                n_freq=args.n_freq, heat_sigma=args.heat_sigma).to(device)
    n_params = sum(p.numel() for p in g.parameters() if p.requires_grad)
    print(f"[g] coord BridgeG depth={args.depth} heads={args.heads} width={args.width} "
          f"n_freq={args.n_freq} heat_sigma={args.heat_sigma} -> {n_params/1e6:.2f}M trainable params")

    opt = torch.optim.AdamW(g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    floor = identity_floor(val_loader, tau, device)
    print(f"[floor] identity (copy-start) changed-cos = {floor:.4f}  (g must beat this via the spec)")

    best_cos, hist = -1.0, []
    for epoch in range(args.epochs):
        run = 0.0
        for b in train_loader:
            zs, zt, spec = b["z_start"].to(device), b["z_goal"].to(device), b["spec"].to(device)
            zp = g.forward_coord(zs, spec)
            loss = bridge_loss(zp, zt, zs, tau, args.lam)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * zs.shape[0]
        train_loss = run / len(train_dset)
        val_loss, val_cos = evaluate(g, val_loader, tau, args.lam, device)
        hist.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_changed_cos": val_cos})
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}: train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| changed-cos {val_cos:.4f}  (Stage-1 gate >= 0.90; floor {floor:.3f})")
        ckpt = {"state_dict": g.state_dict(), "tau": tau, "lam": args.lam,
                "config": {"dim": args.width, "depth": args.depth, "heads": args.heads,
                           "cond_mode": "coord", "n_freq": args.n_freq, "heat_sigma": args.heat_sigma},
                "epoch": epoch, "val_loss": val_loss, "val_changed_cos": val_cos}
        if val_cos > best_cos:
            best_cos = val_cos
            torch.save(ckpt, Path(args.out) / "g_best.pth")
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(ckpt, Path(args.out) / "g_latest.pth")

    # grounding check: swapped-spec changed-cos (should fall toward the identity floor)
    _, swap_cos = evaluate(g, val_loader, tau, args.lam, device, swap_spec=True)
    json.dump({"history": hist, "identity_floor": floor, "best_changed_cos": best_cos,
               "swapped_spec_changed_cos": swap_cos, "tau": tau},
              open(Path(args.out) / "train_history.json", "w"), indent=2)
    print(f"[done] best changed-region cosine = {best_cos:.4f}  "
          f"({'PASS' if best_cos >= 0.90 else 'BELOW'} Stage-1 gate 0.90)")
    print(f"[grounding] correct-spec {best_cos:.4f} vs swapped-spec {swap_cos:.4f} vs identity-floor {floor:.4f} "
          f"(spec is load-bearing if correct >> swapped ~ floor)  -> {args.out}/g_best.pth")


if __name__ == "__main__":
    main()
