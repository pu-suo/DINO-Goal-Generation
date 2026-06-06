# Planning speedup — profile, fix, and result-preservation (masked α=0 / N1)

> SPEED task. Cardinal rule: **do not change planning behavior or success rates.**
> The energy, mask logic, CEM math, success criterion, and the budget defaults
> (`opt_steps`, `num_samples`, `max_iter`, `goal_H`) are unchanged. We optimize
> *execution*, not *semantics*. Validated anchor: masked α=0 (N1/N2) = **0.8 SR @ n=10**
> (`docs/MASKED_ENERGY_RESULTS.md`).

## STEP 1 — Profile (static analysis + runnable profiler)

I read `plan.py`, `plan_multicolor.py`, `planning/{cem,mpc,evaluator,objectives}.py`,
and `models/visual_world_model.py` before touching anything. Two findings up front,
both correcting the task's prior hypotheses:

**(1) The CEM scoring loop is ALREADY batched, ALREADY `no_grad`, ALREADY GPU-resident.**
- `planning/cem.py` builds all `num_samples=300` candidates as one tensor (`repeat` →
  `wm.rollout` advances all 300 in lockstep over `goal_H`) — *not* a per-candidate loop.
- The rollout is already inside `with torch.no_grad()`.
- Elite selection / refit (`torch.argsort`, `mean`, `std`) run on-device; the only host
  sync is one `.item()` per traj for the loss log. So tasks' hypotheses "#1 unbatched"
  and "#3 missing no_grad" did **not** apply; "#4 GPU-resident" was already satisfied.

**(2) The real inner-loop waste: the start observation is re-encoded for every candidate.**
`wm.rollout(obs_0, act)` begins with `self.encode(obs_0)` → `encode_obs` → **DINOv2
`forward` on all `num_samples` copies of the identical start frame**. The *goal* latent
is cached once (`cem.py`), but the *start* latent is recomputed every time. Per MPC
iteration:

    opt_steps(30) × n_evals(10) × num_samples(300) = 90,000 ViT-S/14 forwards
    ...all of just 10 distinct, unchanging images.

That is the redundant work the task's STEP-2 item #2 points at ("remove any per-candidate
re-encoding from the inner loop; only the start observation is encoded once").

**Runnable profiler:** `scripts/profile_plan.py` instruments one episode and prints a
per-category wall breakdown — (a) `predict` (dynamics), (a′) `encode_obs` (DINOv2),
(b) `env.rollout` (pymunk + render), (c) `decode_obs` (VQ-VAE for debug plots),
(d) `objective` (CEM scoring), (e/f) remainder (bookkeeping/transfer/python). Run on the
box:

    CKPTS=/ckpts DATASET_DIR=/data python scripts/profile_plan.py            # quick slice
    CKPTS=/ckpts DATASET_DIR=/data python scripts/profile_plan.py --opt-steps 30
    CKPTS=/ckpts DATASET_DIR=/data python scripts/profile_plan.py --no-fast-encode  # original path

Expected shape (confirm with the run): with the **original** path the redundant DINOv2
encode is a large slice of CEM compute; at the default `eval_every=1` the per-opt-step
`PlanEvaluator` (pymunk rollout + VQ-VAE decode + PNG dump, ×30 per MPC iter) is the other
heavy contributor, of which the **decode + PNG dump is pure visualization**.

## A correctness constraint I found (governs what is/ISN'T safe)

The dynamics model is **never put in `.eval()`** during planning (`plan.py` / `plan_multicolor.py`
load it and leave it in the default `train()` mode), and the predictor ViT config has
`dropout: 0.1` (`conf/predictor/vit.yaml`). **So planning rolls out with active dropout** —
a seeded but stochastic CUDA RNG stream — and the validated 0.8 SR was produced under it.

Consequences (verified in `scripts/test_rng_invariance.py`, runs on CPU):
- The frozen **DINOv2 encoder and proprio encoder draw NOTHING from the RNG stream**
  (dropout p=0 / drop_path=0). The **only** RNG consumer in a rollout is the predictor's
  dropout. → **Caching/hoisting the encode cannot change the dropout masks** → predicted
  latents are identical → CEM scores/elites identical. *This is why the fix is safe.*
- Candidates are drawn with `torch.randn(...)` on the **CPU** generator (no `device=`),
  independent of the CUDA dropout stream. The fix leaves that `randn` call byte-for-byte
  in place → **identical candidates**.
- **Two things would change the RNG stream and are therefore OFF-LIMITS** for the
  result-preserving fix: (i) calling `.eval()` (removes dropout entirely), and
  (ii) batching the `n_evals` loop into one `[n_evals*num_samples,...]` call (changes the
  dropout draw from 10×`batch=300` to 1×`batch=3000`). Both change SR. See "Follow-up".

## STEP 2 — The fix (applied)

All result-preserving, by the RNG argument above.

1. **Cache the start-obs encode** (`planning/cem.py`). Encode `z_obs_0 = encode_obs(obs_0)`
   ONCE per `plan()` (under `no_grad`), broadcast per-traj, and roll out from it via a new
   `VWorldModel.rollout_from_zobs(z_obs_0, act)` that is identical to `rollout(obs_0, act)`
   minus the DINOv2 call. `encode()` was refactored to expose its concat half as
   `_assemble_z()` so `rollout_from_zobs` reuses the exact arithmetic. The per-traj
   `torch.randn` candidate draw is unchanged. Reduces redundant DINOv2 forwards from
   ~90,000 → ~10 per MPC iteration.
   - A `fast_encode` flag (default **on**) keeps the original per-candidate re-encode path
     for A/B regression: `+planner.sub_planner.fast_encode=false`.
   - Numerics: the fast path encodes the start at `batch=n_evals` once vs the original's
     `batch=num_samples`; that is a ~1e-6 cuBLAS kernel-order difference in the start
     latent only (RNG-identical). Scores match to ~1e-4, SR unchanged — verified by
     `scripts/bench_plan.py`.

2. **Skip the inner-eval decode + plot** (`planning/cem.py` + `planning/evaluator.py`).
   `eval_actions(..., plot=False)` for the per-opt-step CEM eval suppresses the VQ-VAE
   decode and PNG dump. `successes`/`logs` are computed *before* plotting (so the
   early-break and SR are untouched), and the decoder is RNG-free (the VQ quantizer's
   `torch.randn` is in `__init__`, not `forward`), so the predictor's dropout stream for
   subsequent opt-steps is unchanged. The final/MPC-outer eval still plots (`plot=True`).

3. **`no_grad` around the hoisted encodes** — the rollout was already `no_grad`; the new
   start/goal encodes are wrapped too. (Did **not** switch to `.eval()` or `inference_mode`:
   the former changes results; the latter risks inference-tensor interaction errors for a
   negligible gain.)

What is deliberately **not** changed: the energy/`objective_fn`, the manipulator mask,
the CEM sampling/topk/refit, the success criterion, `eval_every`'s effect on the
success-based early-break, the env rollout (kept — it feeds that early-break), and all
budget defaults.

## STEP 3 — Regression (run on the box)

`scripts/bench_plan.py` runs the SAME seeds/budget through **both** paths in one process
and asserts SR identical, per-seed pass/fail identical, CEM scores within tol, chosen
actions within tol — then reports the per-episode speedup.

    CKPTS=/ckpts DATASET_DIR=/data python scripts/bench_plan.py          # quick equivalence slice
    CKPTS=/ckpts DATASET_DIR=/data python scripts/bench_plan.py --full   # full validated budget (n=10)

`scripts/test_rng_invariance.py` (CPU, no GPU) independently proves the RNG-neutrality the
fix relies on. **Acceptance**: `--full` must show the masked α=0 SR unchanged vs the
original path.

## eval_every and the residual floor

At the validated `eval_every=1`, the per-opt-step `PlanEvaluator.env.rollout` (pymunk +
render) is retained because it computes the `successes` used by the early-break — removing
it could change SR, so it stays. Its cost is a floor at `eval_every=1`. Phase-0 already
established `eval_every=999` (one SR per MPC iter) is **plan-quality-neutral**
(`docs/PHASE0_ISOLATION_HANDOFF.md`); the **fast-iteration** config below uses it to
collapse that floor. We do **not** change the default `eval_every` for the regression.

## STEP 4 — Fast-iteration config + budget knee (separate from the speed fix)

- `analysis/run_fast_iter.sh` — single masked α=0 condition, `OPT_STEPS`/`NUM_SAMPLES`/
  `N_EVALS` overridable, `eval_every=999`, `fast_encode=true`. For cheap iteration only.
- `analysis/sweep_budget.sh` — sweeps `opt_steps∈{10,15,20,30} × num_samples∈{100,200,300}`
  at n=10 and prints an SR table → pick the smallest budget where SR plateaus (the knee).

Reducing the budget is an iteration accuracy↔speed trade, **not** part of the
regression-tested fix (which uses the full budget).

## Recommended follow-up (opt-in, NOT applied — needs re-validation)

The single biggest remaining lever is **running the dynamics model in `.eval()`** during
planning (disable the predictor dropout). It is almost certainly the intended behavior,
would make rollouts deterministic, and would unlock batching the `n_evals` loop into one
GPU call (≈10× fewer launches) — plausibly the path to "seconds, not minutes". But it
**changes the validated SR** (removes the dropout the 0.8 number was measured under), so it
must be a separate, flagged change with its own re-validation, not folded into this
result-preserving fix.
