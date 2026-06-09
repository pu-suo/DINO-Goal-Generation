"""Train the bridge `g` on cached multi-color latents (Phase 1, specs/G_ARCHITECTURE.md §9 step 4).

Isolated from the dynamics train.py (frozen-everything-else). Data = the Phase-0.2
PushTMultiColorLatentGoalDataset: (z_start, z_goal, instruction) from cached DINOv2 latents.

Frozen text tokens are PRE-ENCODED ONCE into a per-instruction table (never re-encode in the hot
loop). Loss = weighted-L2 to enc(o_goal), up-weighted on changed patches (tau from data). Logs the
Stage-1 fidelity metric (changed-region cosine vs enc(o_goal)); the >=0.90 gate is checked before
wiring g into CEM.

Box (real text encoder):
  python train_bridge.py --latent_dir $DATASET_DIR/pusht_multicolor/latents \
    --data_path $DATASET_DIR/pusht_multicolor --out outputs/bridge/g0 --epochs 100
Local smoke (no transformers): see models/_smoke_train_bridge.py (uses --dummy_text).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.pusht_multicolor_dset import load_multicolor_latent_goal
from models.bridge import (BridgeG, bridge_loss, changed_region_mask, estimate_tau,
                           DIM, N_PATCHES)


class DummyTextEncoder:
    """Deterministic per-instruction random token table -- DEV/SMOKE ONLY (no transformers).

    Same string -> same tokens (so text stays load-bearing and reproducible); different strings
    -> different tokens. NOT a substitute for the real frozen MiniLM at train time on the box.
    """

    def __init__(self, d_text=DIM, max_len=16):
        self.d_text = d_text
        self.max_len = max_len

    def __call__(self, texts):
        toks, masks = [], []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            gen = torch.Generator().manual_seed(seed)
            n_tok = min(self.max_len, 3 + len(t.split()))
            toks.append(torch.randn(self.max_len, self.d_text, generator=gen))
            m = torch.zeros(self.max_len, dtype=torch.bool)
            m[:n_tok] = True
            masks.append(m)
        return torch.stack(toks), torch.stack(masks)


def build_text_table(instructions, encoder, chunk=64):
    """Encode every UNIQUE instruction once -> {instruction: (tokens (L,d), mask (L,))} on CPU."""
    uniq = sorted(set(instructions))
    table = {}
    for i in range(0, len(uniq), chunk):
        c = uniq[i:i + chunk]
        tk, mk = encoder(c)
        for j, instr in enumerate(c):
            table[instr] = (tk[j].detach().cpu(), mk[j].detach().cpu())
    print(f"[text] cached {len(table)} unique instructions (tokens {tuple(tk.shape[1:])})")
    return table


def collate(items):
    return {
        "z_start": torch.stack([it["z_start"] for it in items]),
        "z_goal": torch.stack([it["z_goal"] for it in items]),
        "instruction": [it["instruction"] for it in items],
    }


def gather_text(instructions, table):
    tk = torch.stack([table[i][0] for i in instructions])
    mk = torch.stack([table[i][1] for i in instructions])
    return tk, mk


def changed_region_cosine(pred, target, z_start, tau):
    """Mean cosine(pred, target) over CHANGED patches -- the Stage-1 fidelity metric (§8)."""
    changed = changed_region_mask(z_start, target, tau)            # (B,196)
    cos = F.cosine_similarity(pred, target, dim=-1)                # (B,196)
    return (cos * changed).sum() / changed.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(g, loader, table, tau, lam, device):
    g.eval()
    tot, cosw, n = 0.0, 0.0, 0
    for b in loader:
        zs, zt = b["z_start"].to(device), b["z_goal"].to(device)
        tk, mk = gather_text(b["instruction"], table)
        zp = g(zs, tk.to(device), mk.to(device))
        bs = zs.shape[0]
        tot += bridge_loss(zp, zt, zs, tau, lam, reduction="none").sum().item()
        cosw += float(changed_region_cosine(zp, zt, zs, tau)) * bs
        n += bs
    g.train()
    return tot / max(n, 1), cosw / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", required=True, help="<latent_dir>/<split>/{start,goal}_latents.pth")
    ap.add_argument("--data_path", required=True, help="<data_path>/<split>/labels.pkl")
    ap.add_argument("--out", default="outputs/bridge/g0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--width", type=int, default=DIM)
    ap.add_argument("--lam", type=float, default=7.0, help="changed-patch up-weight (§4 lambda)")
    ap.add_argument("--tau", type=float, default=None, help="override; default = Otsu on train set")
    ap.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--text_max_len", type=int, default=16)
    ap.add_argument("--dummy_text", action="store_true", help="DEV: deterministic random text tokens")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out, exist_ok=True)

    dsets = load_multicolor_latent_goal(args.latent_dir, args.data_path, splits=("train", "val"))
    train_dset, val_dset = dsets["train"], dsets["val"]
    print(f"data: train={len(train_dset)} val={len(val_dset)} | device={device}")

    # frozen text encoder (real MiniLM on the box; dummy for local smoke)
    if args.dummy_text:
        encoder = DummyTextEncoder(d_text=args.width, max_len=args.text_max_len)
        d_text = args.width
        print("[text] DUMMY text encoder (dev only)")
    else:
        from models.bridge import FrozenTextEncoder
        encoder = FrozenTextEncoder(args.text_model, max_len=args.text_max_len, device=device)
        d_text = encoder.d_text
        print(f"[text] frozen {args.text_model} (d_text={d_text})")

    all_instr = [train_dset.labels[i]["instruction"] for i in range(len(train_dset))] \
        + [val_dset.labels[i]["instruction"] for i in range(len(val_dset))]
    table = build_text_table(all_instr, encoder)

    # tau from the train set (§4: histogram valley); log the changed-patch fraction
    tau = args.tau if args.tau is not None else estimate_tau(train_dset.start, train_dset.goal)
    frac = changed_region_mask(train_dset.start, train_dset.goal, tau).mean().item()
    print(f"[tau] tau={tau:.3f} -> {frac*100:.1f}% of patches flagged changed (train)")

    g = BridgeG(dim=args.width, depth=args.depth, heads=args.heads, d_text=d_text).to(device)
    n_params = sum(p.numel() for p in g.parameters() if p.requires_grad)
    print(f"[g] BridgeG depth={args.depth} heads={args.heads} width={args.width} "
          f"-> {n_params/1e6:.1f}M trainable params")

    opt = torch.optim.AdamW(g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    best_cos, hist = -1.0, []
    for epoch in range(args.epochs):
        run = 0.0
        for b in train_loader:
            zs, zt = b["z_start"].to(device), b["z_goal"].to(device)
            tk, mk = gather_text(b["instruction"], table)
            zp = g(zs, tk.to(device), mk.to(device))
            loss = bridge_loss(zp, zt, zs, tau, args.lam)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * zs.shape[0]
        train_loss = run / len(train_dset)
        val_loss, val_cos = evaluate(g, val_loader, table, tau, args.lam, device)
        hist.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_changed_cos": val_cos})
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}: train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| changed-cos {val_cos:.4f}  (Stage-1 gate >= 0.90)")
        ckpt = {"state_dict": g.state_dict(), "tau": tau, "lam": args.lam,
                "config": {"dim": args.width, "depth": args.depth, "heads": args.heads, "d_text": d_text},
                "text_model": (None if args.dummy_text else args.text_model),
                "epoch": epoch, "val_loss": val_loss, "val_changed_cos": val_cos}
        if val_cos > best_cos:
            best_cos = val_cos
            torch.save(ckpt, Path(args.out) / "g_best.pth")
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(ckpt, Path(args.out) / "g_latest.pth")

    json.dump(hist, open(Path(args.out) / "train_history.json", "w"), indent=2)
    print(f"[done] best changed-region cosine = {best_cos:.4f}  "
          f"({'PASS' if best_cos >= 0.90 else 'BELOW'} Stage-1 gate 0.90)  -> {args.out}/g_best.pth")


if __name__ == "__main__":
    main()
