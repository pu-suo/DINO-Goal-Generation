# Ruled out — negative results log

Things we tried and decided NOT to carry forward. Recorded so they are not silently
re-attempted. Each entry: what, the evidence, the root cause, and what was kept/removed.

---

## Learned quasimetric cost-to-go (IQE/QRL) over masked DINO latents  — RULED OUT (2026-06)

**What:** a QRL-style **pure-V\*** quasimetric cost-to-go `d_θ(z_a, z_b)` (IQE-maxmean head,
no transition/Q/policy) over the **masked object-only** DINO-WM latents, added to the CEM
energy as a dense, asymmetric, long-range shaping term:
`energy = w_l2 · masked-L2 + w_qm · d_θ (+ α · proprio)`. The hypothesis: an asymmetric
learned cost-to-go would close the gap between the masked-L2 **floor** (SR≈0.80, the α=0
object-only energy) and the real-pusher proprio **ceiling** (SR≈0.97) on genuine single-T PushT.

**Result — it did NOT beat the floor.** On genuine held-out single-T goals, regular budget, n=30:
- FLOOR (masked-L2, α=0)  = **0.80**
- NEW (masked-L2 + quasimetric, w_qm=1, w_l2=10) = **0.73**  ← below the floor
- CEILING (real pusher, α=1) = **0.97**

The first head (`iqe_d0`) **failed its validation gates by silent overfitting** (held-out
adjacent-d ≈9× train; monotonicity 0.54 < 0.7; `d_sg` inflated to ~108 from too-soft φ).
A recalibrated head (`iqe_d1`: sharper φ β 0.1→0.5, bounded goal offsets, more data, fewer
steps, weight decay, best-val-on-monotonicity checkpointing) fixed the *level* metrics
(`d_sg`≈30–45) but the cost-to-go remained **globally monotone yet locally noisy**
(validator: decreasing-frac 0.63, Spearman 0.74 — a *representational* shortfall, not an
optimization one). Its SR (0.73) still did not beat 0.80.

**Root cause (attributed):** the **frozen masked DINO latent's pose signal is jittery at fine
scale** — the same noise the value head inherits. CEM ranks candidates by argmin of the energy,
so locally-noisy ordering is exactly what breaks; a learned scalar can't add ordering that the
representation doesn't carry. (Whether this jitter is real and how large is now measured
directly by `analysis/pose_decode_probe.py` — see below.)

**Decision:** drop the quasimetric. The Phase-0 gate (oracle ≥0.80) is already met by free
masked-L2, so the high-effort head bought a *negative* marginal return. Effort redirected to
the pose-decode probe, which decides whether the **representation itself** can support `g`.

**Removed (recoverable from git history; see the commit that deletes them):**
`models/quasimetric.py`, `datasets/qm_latent_dset.py`,
`scripts/{train,test}_quasimetric.py`, `scripts/relaunch_qm.sh`,
`analysis/run_qm_eval.sh`, `analysis/validate_quasimetric.py`,
`docs/QUASIMETRIC_RUNBOOK.md`, and the vendored `third_party/torchqmet/`.
The qm branch was pruned from `plan.py`, `planning/objectives.py`
(`create_qm_objective_fn`), and `conf/plan_pusht.yaml` (`qm:` block) — the masked-L2 floor
path is untouched.

**Kept (NOT quasimetric-specific):** the manipulator masking
(`env.pusht.multicolor_common.manipulator_energy_mask`), the masked-L2 / floor energy
(`planning.objectives.create_objective_fn` → `objective_fn_last`), the CEM speedup, and the
trajectory-latent cache script (renamed `scripts/cache_qm_latents.py` →
`scripts/cache_traj_latents.py`) which the pose-decode probe consumes.

**Follow-up — masked-DINO pose-decode probe outcome (RAN 2026-06, pusht_noise, 6000 trajs,
16k train / 8k test frames, whole-traj split):** verdict = **g_viable**. The masked latent
DECODES pose cleanly, so the quasimetric's failure was SEARCH/VALUE-fitting, NOT the
representation — confirming hypothesis (A), refuting (B).
- masked orientation MAE: **linear 4.4° (median 2.4°)**, MLP 5.1° — 97% of frames < 20°.
- masked position: **5.4px L2 (median 3.6)** — 97% < 20px; "both gates" satisfied on **96%** of
  frames (a decodability ceiling ABOVE the 0.80 masked-L2 floor; the planning-SR gap is the
  planner/dynamics, not the latent).
- ACCESS: linear ≈ MLP -> pose is CLEANLY LINEARLY accessible (a simple `g` head suffices).
- PUSHER: unmasked − masked = +0.0° -> the object-only latent carries pose on its own; masking
  the pusher costs ~no pose info (validates the masked-energy design).
- SMOOTHNESS: residual jitter 3.2° (« the ~9°/step true rotation) -> NO pathological jittery
  fine-pose. The jitter the quasimetric inherited is in the VALUE FIT, not the representation.
- => Representation is NOT the bottleneck; build `g` on this latent (Phase 1). Remaining risk
  shifts to `g`'s synthesis + CEM control, not the latent. (Decode transfers to multicolor —
  same frozen DINOv2 — but the multicolor oracle SR ≥0.80 held-out is still a separate measure.)
