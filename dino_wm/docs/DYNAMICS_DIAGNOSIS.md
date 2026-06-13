# Multicolor Dynamics Diagnosis — 2026-06-11

> Diagnose-then-fix for the Stage-0 blocker: M-real (real dset goal, easiest energy)
> = **0.40 pose-SR** (n=10) vs the 0.80 gate, capping `g`. All numbers below are
> **measured** on the vast.ai 4090 unless marked *reasoned*.

## FINAL VERDICT (2026-06-12): REPRESENTATIONAL CEILING — not data, epochs, recipe, or phase-aug

> The "DATA-BOUND" verdict below (kept for the record) was the *initial* hypothesis. After
> building a cached-latent fast-training path and exhausting every dynamics-side lever, the
> conclusion flipped: the multicolor block-region latent is **~2× harder to predict than
> single-T regardless of training**, because the frozen DINOv2 ViT-S contextualizes the
> block's patch tokens with the 4 colored decals (ViT attention) → a representational floor
> the dynamics predictor cannot train through. Same root as the pose-decode anomaly (§4).

**block tf-1-step floors at ~16 (stock single-T = 8.0); all four levers exhausted:**

| lever | result | block tf-1step |
|---|---|---|
| **Data** (2k→5k→10k, warm-start) | flat | 17.2 → 16.4 → 15.9 |
| **Epochs** (grid-10k to convergence, ep17→45) | flat | ~16 (+ M-real SR flat 0.4) |
| **Phase-aug** (stride-1-2k vs grid-2k, matched, converged) | +0.6 only | 16.4 vs 17.0 |
| **Recipe-fidelity** (stride-1 reproduces train.py exactly) | confirmed | — |
| stock single-T (own data, plans @0.90) | the floor | **8.0 / 7.6** |

- Converged grid-10k **M-real SR = 0.40** = the 2k baseline ⇒ full training budget did not
  improve *planning*, the metric that matters (not just block-tf).
- grid-2k overfits (train 0.11 ≪ val 0.21) yet block-tf stays 17 ⇒ NOT capacity- or
  data-limited; it generalizes to the best predictor the contaminated latent allows.
- Data curve is **flat, not slowly-improving** ⇒ not on a data-scaling curve; more data
  (18k/24k) would not help. **24k gen killed.**

**The lever is NOT dynamics training — it is the scene/encoder representation.** Candidate
fixes (next decision): fewer decals (2 vs 4), thinner/subtler decal rendering, or a larger
frozen encoder (DINOv2 ViT-B/L) that separates block from decals better. Confirm-the-mechanism
experiment: decal-count ablation (1/2/4 decals → block-tf); if 1-2 decals → block-tf ~8-10,
the contamination mechanism is proven and the testbed sweet-spot is found.

Tooling built this session (all on `main`): cached-latent dynamics training
(`scripts/cache_dynamics_latents.py` + `datasets/dyn_latent_dset.py`
{DynLatentSliceDataset, StrideOneLatentDataset} + `train_dynamics_cached.py`,
forward_latent in `models/visual_world_model.py`) — ~16h/epoch → minutes; mmap load,
on-GPU loss accum; `tests/test_{forward_latent_equiv,stride_one_windows,bridge_override}.py`.

---

## (Superseded) Initial verdict: DATA-BOUND

The retrained multicolor dynamics (`outputs/2026-06-09/23-16-24`, predictor-only,
`num_pred=1 f5 h3`, 9 epochs on 2,000 trajs) is ~**2× worse than stock in BOTH
prediction regimes with a stock-normal compounding rate** — the signature of
under-fitting, not exposure bias. The earlier "exposure bias / rollout-aware recipe
needed" conclusion came from comparing our tf-1-step to stock's **free-rollout**
baseline (14.5/12.2) instead of stock's tf-1-step (8.0/7.6) — a regime mix-up.
**This was wrong** — see the final verdict above; the data fix did not move block-tf.

## Evidence

### 1. dynamics_check, ours vs shipped (n=50 traj, H=10, epoch-9 `model_latest`)

| block/pusher latent-L2 | tf-1step | free-roll H=10 | growth |
|---|---|---|---|
| OURS, multicolor test | 17.2 / 16.3 | 27.1 / 27.9 | 1.6–1.7× |
| OURS, multicolor train | 13.6 / 14.5 | 24.3 / 26.9 | 1.8× |
| STOCK on multicolor test (OOD) | 29.0 / 26.6 | 42.8 / 42.5 | 1.5–1.6× |
| STOCK on pusht_noise val (own data) | **8.0 / 7.6** | **14.5 / 12.2** | 1.6–1.8× |

JSONs: `analysis_outputs/dynchk_{ours_test,ours_train,stock}.json`.
- tf-1-step already ~2.1× stock ⇒ under-trained (the brief's discriminator).
- Identical tf→free growth ⇒ no recipe-specific rollout pathology.
- train≈test (1.1–1.26×) ⇒ not overfitting; train error itself ≫ stock ⇒ under-fit.

### 2. Data scale (the cause)

- Stock pusht: `n_rollout: None` ⇒ **all 18,685 trajs / 1.98M slices** (mean len 125).
- Ours: **2,000 trajs / 162k slices** (T=100) ⇒ **12.2× fewer samples**, on a HARDER
  scene (4 random decals vs 1 fixed goal-T). Same recipe (`num_pred=1 h3 f5`,
  normalize_action, frozen encoder) ⇒ the recipe demonstrably produces a 0.90-planning
  model at stock scale.

### 3. Confounds cleared

- **Stats (2a): CLEAR.** Train normalized with multicolor `stats.pth`
  (`normalize_action=True` + `_load_stats` in `datasets/pusht_multicolor_dset.py`);
  the 0.40 M-real run used `multicolor.stats_source=multicolor` (command recovered
  from tmux scrollback; identical overrides in `plan_outputs/20260611040750_*`,
  `model_epoch=8`, fast bundle, `goal_H=5`, `max_iter=10`; `logs.json` mtime 04:48
  matches the ~41-min fast run).
- **Overshoot/horizon (2b): NOT the cap.** M-real with `goal_H=2 max_iter=20`
  (same fast bundle/seed/epoch) = **0.5** (vs 0.40); goal_H also shortens dset goals
  (frameskip·goal_H), so these were 10-env-step trivial pushes — and the model still
  failed half. Failure geometry is **bimodal**: successes 0–25px, failures
  **92–334px** with 0.4–1.2 rad error and ~0 coverage (gross wrong-direction
  failures, not near-miss oscillation). gH3 (goal_H=3, same knobs) = **0.5** (coverage 0.45) -- the full horizon response is 0.40 (gH5) -> 0.50 (gH2) -> 0.50 (gH3): <= +0.1 from config, noise-level at n=10.
- **Combo split (2c-C): CLEAR.** Pose-decode fit on train-combos vs held-out combos:
  31.7px vs 33.7px — combo recombination costs ~2px, not the story.

### 4. Multicolor pose-decodability (separate g-side flag, NOT the dynamics cap)

Linear (dual-ridge λ=10, masked) pose decode, vs the single-T reference
**5.4px/4.4°/96%-in-gate at n_fit=16,000** (`pose_decode_probe.json`):

| fit set → eval set | pos MAE | θ MAE | in-gate |
|---|---|---|---|
| 2,000 goal frames → 400 test goals | 32.0px | 30.9° | 0.16 |
| 2,000 start frames → 400 test starts | 37.1px | 40.1° | 0.12 |
| 500/1000/2000 goals (slope) | 41→37→32px | — | 0.06→0.16 |
| 16k arbitrary frames → 2k test frames | **TBD_SCALED** | | |

Start frames decode no better than goals ⇒ NOT a block-on-decal-overlap effect.
The n_fit slope was still falling at 2k. The 16k like-for-like fit (+ a no-mask
variant: mid-push the pusher abuts the block, so masking deletes block-adjacent
patches — a confound unique to mid-trajectory frames) reruns after the GPU frees:
**TBD_SCALED_VERDICT**. Note CEM plans on latent-L2, not a linear pose readout —
this flags g's grounding-metric reliability and latent separability, it does not
gate the dynamics fix.

## Fix (approved at gate): data scaling, same recipe

1. **Regen at 10k** (`scripts/gen_pusht_multicolor.py --n_train 10000 --seed 0`):
   per-episode seeds are `base+i` ⇒ strict superset of the 2k set; val/test
   reproduce; combo split frozen. Measured throughput ≈5.4 eps/s @12 workers ⇒
   ~30 min. → `/workspace/data/pusht_multicolor_10k` (+ recomputed stats.pth).
2. **Retrain predictor-only, warm-started** from epoch-9 (`model_latest.pth` copied
   into the run dir; train.py resumes at epoch 10), `training.epochs=13` ⇒ 4 epochs
   on 10k. Encoder stays frozen. *Reasoned* cost: 2h45/epoch at 2k scales ~linearly
   (per-step DINOv2 fwd + mp4 decode dominate) ⇒ ~14h/epoch, mitigated by
   `env.num_workers=24` + `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (env-var-only TF32).
   Run dir: `outputs/2026-06-11/retrain10k`.
3. **Per-epoch tracking:** CPU dynamics_check (n=20) after each epoch ckpt
   (`analysis_outputs/dynchk_10k_ep*.json`); expect free-roll block/pusher to move
   from ~27/28 toward stock's ~14.5/12.2.
4. **Escalation if needed** (data-equivalent compute exhausted): more data (18k+)
   before any recipe change; k-step unrolled loss only if generous data scaling
   leaves free-rollout ≫ stock with tf-1-step ≈ stock (the true recipe signature).

## Verification ladder (after retrain)

1. dynamics_check GPU n=50 H=10 with `--stats pusht_multicolor_10k/stats.pth`.
2. M-real n=10 (fast bundle, `multicolor.data_path=...10k`), then n=30 if ≥~0.7.
3. Named-target oracle, held-out combos, deployable energy
   (`goal_source=named_target multicolor.goal_pusher=contact objective.alpha=1`, and
   the masked variant `use_manipulator_mask=true objective.alpha=0`): gate ≥0.80.
4. g Stage-2 closed-loop (`goal_source=bridge`, new seam in `planning/cem.py`
   `z_obs_g_override` + `plan_multicolor._attach_bridge_goal`): success = env pose
   check vs the NAMED target with g's synthesized goal latent; floors = swapped-text
   + instruction-agnostic.

## g status under the fix

- Stage-1 (latent fidelity, frozen-everything): PASSED held-out (changed-cos 0.9255
  vs 0.69 floors) — independent of the dynamics; not invalidated.
- g0 was trained on the 2k pairs; after the 10k cache lands, retrain g on 10k pairs
  (`train_bridge.py --latent_dir .../pusht_multicolor_10k/latents`) → `outputs/bridge/g10k`,
  re-run Stage-1 on the same 400 held-out episodes for comparability.
- Stage-2 closed-loop was BLOCKED on the dynamics cap (g inherits the planner's
  model); it becomes meaningful once M-real clears.
