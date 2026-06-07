# Masked-DINO pose-decode probe — the go/no-go for `g`

**Question it answers:** does the **frozen, masked (object-only) DINO-WM latent** actually
*contain* the block's pose (x, y, θ) that `g` must synthesize and the CEM planner must
reach? This is the decider for whether to build `g` on this representation now, or whether
a higher-resolution representation (e.g. V-JEPA-2-AC) is on the critical path first.

It exists because the learned quasimetric cost-to-go **did not beat the masked-L2 floor**
(floor SR=0.80, qm SR=0.73, n=30). Two explanations: (A) the value head couldn't fit a
clean cost-to-go though the pose info is present → `g` is viable; (B) the masked latent's
own pose signal is too noisy → no value function and no `g` can win. The probe decides A vs B
by **directly decoding** pose from the latent and comparing the decode error to the success
gate (`pos < 20` sim-px, `angle < π/9 = 20°`).

## Run it (vast.ai, ~1 min on a 4090)

```bash
cd /workspace/dino_goal/dino_wm && source $WS/activate.sh

# Option A — REUSE the existing cache from the quasimetric run (NO re-cache):
python analysis/pose_decode_probe.py --cache_dir $DATASET_DIR/pusht_noise/qm_latents

# Option B — fresh instance / cache gone: regenerate the trajectory-latent cache (~4 min),
# then run with the default --cache_dir (.../pusht_noise/traj_latents):
python scripts/cache_traj_latents.py --splits train --n_rollout 6000
python analysis/pose_decode_probe.py
```

The probe **reuses cached latents** (no re-encoding) and is read-only on the frozen
encoder / dynamics / planner / mask. Output → `analysis_outputs/pose_decode_probe/`:
`pose_decode_probe.json` (all metrics + the printed diagnosis) and four plots
(`theta_scatter`, `xy_scatter`, `theta_time`, `err_hist`).

## What it does

- **MAIN:** decode block pose from the **masked object-only** latent (pusher patches zeroed
  with the planner's `manipulator_energy_mask` — the exact mask the energy uses).
- **CONTROL 2 (pusher contribution):** decode the same pose from the **full unmasked** latent;
  report `unmasked − masked` orientation error. A large gap ⇒ the pusher was carrying the pose.
- **CONTROL 3 (smoothness):** decode θ over time on held-out trajectories (with the overfit-free
  **linear** decoder) and report the **residual jitter** — std of `d(unwrap(decoded)) −
  d(unwrap(true))`, the frame-to-frame decode-error change (isolates decode jitter from the true
  rotation-speed variability). Direct test of the "jittery fine-pose" hypothesis.
- Two probes per condition: a **linear** ridge (on the full latent) and a **2-layer MLP on
  PCA-reduced features** (a wide net over the raw 75264-dim grid overfits and can't beat the
  linear probe). The linear↔MLP gap shows whether pose is cleanly accessible or present-but-entangled.
- **Trajectory split** (whole trajs held out — never a frame split, which overstates decodability).
- Angle handled as `(sin, cos)` → `atan2` → wrapped minimal difference; the PushT T is **C1
  (no rotational symmetry)** so the full `[0, 2π)` range applies (no folding). Position error is
  the **2D Euclidean** norm in sim-512 px — directly comparable to the 20-px gate.

## How to read the diagnosis (it prints one; keyed off the masked latent)

(keyed off the masked orientation MAE of the **better** of the two decoders)

| masked orientation MAE | verdict |
|---|---|
| **< ~15°** | pose **IS present** → quasimetric failure was search/value, not representation → **`g` is viable** |
| **~15–30°** | **borderline** — fine for the loose 20° gate on most goals, jittery on hard rotations (consistent with the observed 0.80); **25–30° leans toward the ceiling** |
| **> ~30°** | **hard representational ceiling** → a higher-resolution representation is on the critical path **before** `g` |

Extra flags the diagnosis raises:
- If orientation MAE **> 20°**: the representation's own pose noise exceeds the success
  criterion — no planner can reliably hit the target regardless of the value function.
- **PUSHER:** if unmasked decodes orientation much better than masked, the pusher was the anchor.
- **ACCESS:** linear≈MLP ⇒ cleanly (linearly) accessible; MLP≫linear ⇒ present-but-entangled.
  (The verdict uses the *better* of the two decoders, so an undertrained MLP can't force a
  false "ceiling".)

## Decision

- **`g_viable` / borderline** → proceed to Phase 1 (build `g`) on the masked DINO latent.
- **representation_ceiling** → do **not** build `g` on this latent yet; a higher-resolution
  representation (V-JEPA-2-AC) becomes the next step.

Append the probe's printed verdict to `docs/RULED_OUT.md` once known (it closes the
quasimetric thread there).

## Local smoke (Mac, no real data)

`python analysis/_smoke_pose_decode.py` synthesizes a tiny cache whose latents linearly
encode pose and asserts the linear probe recovers it near-perfectly — i.e. the angle math,
masking, trajectory split, and gate comparison are wired correctly.
