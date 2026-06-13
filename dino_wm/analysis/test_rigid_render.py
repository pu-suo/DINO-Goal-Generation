"""Part-1 render guards 1.3 (walls pixel-identical) & 1.5 (green-T zero residual),
each with a should-pass/should-fail, plus a before/after visual.

  /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python analysis/test_rigid_render.py
"""
import os, sys, pickle
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.rigid_goal_render import make_env, render_state, green_pixel_count, border_ring, LIGHTGREEN
from datasets.rigid_goal_common import apply_se2, make_language

DEV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_devdata", "pusht_noise_val")
states = torch.load(os.path.join(DEV, "states.pth")).double().numpy()
seq = pickle.load(open(os.path.join(DEV, "seq_lengths.pkl"), "rb"))

results = []
def check(name, cond):
    results.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

env_clean = make_env(with_target=False)
env_green = make_env(with_target=True)

# ===== GUARD 1.3 -- walls pixel-identical across different poses ==============
print("GUARD 1.3  walls/boundary pixel-identical (render by STATE, not image warp)")
# two interior states (block central, away from the border) -> border is wall+bg only
sa = np.array([256, 256, 230, 270, 0.5]); sb = np.array([256, 256, 280, 240, 2.1])
ia, _ = render_state(env_clean, sa); ib, _ = render_state(env_clean, sb)
ba, _ = border_ring(ia, 6); bb, _ = border_ring(ib, 6)
check("should-PASS: outer wall ring identical between two different-pose frames",
      np.array_equal(ba, bb))
# should-FAIL control: if we'd AFFINE-WARPED the image the walls would move. Emulate by
# rolling frame B 4px -> the border ring now differs (demonstrates the failure we avoid).
ib_warp = np.roll(ib, 4, axis=1)
bbw, _ = border_ring(ib_warp, 6)
check("should-FAIL(detected): an image-warped frame moves the walls -> ring differs",
      not np.array_equal(ba, bbw))

# ===== GUARD 1.5 -- green goal-T removed, zero residual ======================
print("\nGUARD 1.5  green goal-T removal (zero residual, no halo)")
# render a state where BOTH block and pusher are away from the center, so the
# green-T at (256,256) is fully unoccluded (else the pusher/block recolors it).
s_show = np.array([400, 400, 150, 150, 0.3])
ig, _ = render_state(env_green, s_show)     # with_target=True -> green-T at (256,256)
ic, _ = render_state(env_clean, s_show)     # with_target=False -> removed
ng, nc = green_pixel_count(ig), green_pixel_count(ic)
check(f"should-PASS: green-T VISIBLE with target on ({ng} green px > 0)", ng > 0)
check(f"should-PASS: ZERO green residual with target off ({nc} green px == 0)", nc == 0)
# also: the removed region must equal background where not covered by block/pusher.
# Pixels that were green in ig should be white (255) in ic.
green_mask = np.abs(ig.astype(int) - LIGHTGREEN[None, None]).max(2) <= 40
ic_at_green = ic[green_mask]
frac_white = float((ic_at_green == 255).all(axis=1).mean()) if green_mask.sum() else 1.0
check(f"should-PASS: removed green-T region is background-white ({frac_white*100:.1f}% white)",
      frac_white > 0.999)

# ===== before/after visual: a REAL push -> rigid-transformed clean triple =====
# pick a wall-free-ish traj, transform it, render original (green) vs transformed (clean)
i = 2; L = seq[i]
o0, oT = states[i, 0], states[i, L - 1]
th, t = 0.9, np.array([-20.0, 35.0])
t0, tT = apply_se2(o0, th, t), apply_se2(oT, th, t)
lang = make_language(t0, tT)
imgs = {
    "orig start (green-T)": render_state(env_green, o0)[0],
    "orig goal (green-T)":  render_state(env_green, oT)[0],
    "tf start (clean)":     render_state(env_clean, t0)[0],
    "tf goal (clean)":      render_state(env_clean, tT)[0],
}
fig, axes = plt.subplots(2, 2, figsize=(8, 8.8))
for ax, (title, im) in zip(axes.ravel(), imgs.items()):
    ax.imshow(im); ax.set_title(title, fontsize=10); ax.axis("off")
fig.suptitle("Real push (top, with goal-T) -> rigid-transformed clean triple (bottom)\n"
             + lang["text"], fontsize=10)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis_outputs", "rigid_render_demo.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nsaved before/after visual -> {out}")

print("\n" + "=" * 60)
print(f"RENDER GUARD TESTS: {sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
