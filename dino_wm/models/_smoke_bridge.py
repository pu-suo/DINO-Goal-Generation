"""Local CPU smoke for models/bridge.py (the `g` module) -- no data / GPU / network.

Verifies the spec's required unit tests (specs/G_ARCHITECTURE.md §3, §4, §9):
  1. forward shapes: (B,196,384) + text -> (B,196,384).
  2. zero-init gate => z_goal == z_start at init (residual identity).
  3. estimate_tau / changed_region_mask split object vs background.
  4. overfit a tiny TEXT-DEPENDENT batch to ~0 weighted-L2 loss (module + loss wired right).
  5. text is load-bearing: swapping the text changes z_goal.

    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python models/_smoke_bridge.py
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.bridge import (BridgeG, bridge_loss, estimate_tau, changed_region_mask,
                           N_PATCHES, DIM)


def main():
    torch.manual_seed(0)
    B, L, d_text = 4, 8, DIM

    # 1 + 2: shapes and identity-at-init -----------------------------------------------
    g = BridgeG(depth=2, d_text=d_text)
    z_start = torch.randn(B, N_PATCHES, DIM)
    text_tokens = torch.randn(B, L, d_text)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    text_mask[:, 5:] = False                       # pad the last 3 positions
    z_goal = g(z_start, text_tokens, text_mask)
    assert z_goal.shape == (B, N_PATCHES, DIM), f"bad shape {tuple(z_goal.shape)}"
    assert torch.allclose(z_goal, z_start, atol=1e-6), \
        "zero-init gate must make z_goal == z_start at init (residual identity)"
    print("OK 1-2: forward shapes + zero-init identity (z_goal == z_start)")

    # 3: tau + changed-region mask -----------------------------------------------------
    obj = list(range(10))
    z_t = z_start.clone()
    z_t[:, obj, :] += 3.0                           # strong change only in 'object' patches
    tau = estimate_tau(z_start, z_t)
    m = changed_region_mask(z_start, z_t, tau)
    assert m[:, obj].mean() > 0.99 and m[:, 10:].mean() < 0.01, \
        f"changed mask failed to split object/background (tau={tau:.3f})"
    print(f"OK 3: estimate_tau={tau:.2f} splits 10 object patches from 186 background")

    # 4: overfit a tiny TEXT-DEPENDENT batch -------------------------------------------
    g = BridgeG(depth=2, d_text=d_text)
    W = torch.randn(d_text, DIM) * 0.5
    text_emb = (text_tokens * text_mask.unsqueeze(-1)).sum(1) / text_mask.sum(1, keepdim=True)
    delta_obj = text_emb @ W                        # (B, DIM): per-sample, text-driven change
    z_target = z_start.clone()
    z_target[:, obj, :] += delta_obj.unsqueeze(1)   # broadcast the change over object patches
    tau = estimate_tau(z_start, z_target)
    opt = torch.optim.Adam(g.parameters(), lr=2e-3)
    l0 = bridge_loss(g(z_start, text_tokens, text_mask), z_target, z_start, tau).item()
    for _ in range(600):
        opt.zero_grad()
        loss = bridge_loss(g(z_start, text_tokens, text_mask), z_target, z_start, tau)
        loss.backward()
        opt.step()
    lf = loss.item()
    print(f"OK 4: overfit weighted-L2 loss {l0:.4f} -> {lf:.6f}")
    assert lf < l0 * 0.05, f"overfit failed ({l0:.4f} -> {lf:.4f})"

    # 5: text is load-bearing ----------------------------------------------------------
    z_a = g(z_start, text_tokens, text_mask)
    z_b = g(z_start, torch.randn(B, L, d_text), text_mask)
    assert (z_a - z_b).abs().mean() > 1e-3, "z_goal must depend on the text (grounding)"
    print("OK 5: text conditioning is load-bearing (swapping text changes z_goal)")

    print("SMOKE OK: g forward/identity/loss/tau/text-grounding all wired correctly.")


if __name__ == "__main__":
    main()
