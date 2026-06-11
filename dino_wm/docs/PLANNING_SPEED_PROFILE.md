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

> **CORRECTION (2026-06-10): the premise below is almost certainly WRONG.** The claim
> "the model is left in the default `train()` mode" was an inference, not a measurement.
> The code says otherwise: checkpoints pickle **whole module objects**, `train.py` calls
> `save_ckpt()` immediately after `self.val()` (train.py:389-392) whose first line is
> `self.model.eval()` (train.py:550) — recursively setting `predictor.training=False` —
> pickle preserves that flag, and nothing in the plan path ever calls `.train()`. The
> same val-before-save order exists in upstream `dino_wm` at the import commit, i.e. the
> code that produced the shipped OSF `pusht` checkpoint. **Expected reality: the
> predictor was ALWAYS in eval mode at plan time, dropout was never active, and the
> validated 0.8 SR is a dropout-free number.** Verify per checkpoint (30 s, on the box):
>
>     python scripts/check_ckpt_train_mode.py /workspace/ckpts/outputs/pusht/checkpoints/model_latest.pth
>
> If it prints `training=False` for the predictor (expected): every "dropout RNG stream"
> argument in this section is moot — rollouts were already deterministic, the only RNG
> consumer in planning is the CPU `torch.randn` candidate draw in `planning/cem.py`, and
> the changes below in "Fast config bundle" are gated only by ordinary floating-point
> numerics. If it prints `True` (unexpected), the original analysis below stands.

The dynamics model is **never explicitly put in `.eval()`** during planning, and the
predictor ViT config has `dropout: 0.1` (`conf/predictor/vit.yaml`). The original analysis
assumed planning therefore rolls out with active dropout — a seeded but stochastic CUDA
RNG stream — and that the validated 0.8 SR was produced under it. (See the correction
above: the pickled mode, not a default, governs, and it is expected to be eval.)

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
  *(2026-06-10: both are now SHIPPED as opt-in flags — `plan_eval_mode` and
  `planner.sub_planner.traj_chunk` — under the "Fast config bundle" below, and the
  dropout premise behind this OFF-LIMITS reasoning is expected false per the CORRECTION
  above. The constraint stands only for the legacy result-preserving default config.)*

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

## Recommended follow-up (superseded — implemented as the "Fast config bundle" below)

The original recommendation here ("run the model in `.eval()` to disable dropout and
unlock batching") is superseded twice over: (a) the dropout premise is expected false
(see the CORRECTION above — checkpoints are pickled in eval mode), and (b) the bundle
below implements the batching and the rest of the levers, all opt-in.

## Fast config bundle (2026-06-10; opt-in, default OFF = stock behavior)

Implemented per the 2026-06-10 cost audit (20-agent adversarially-verified pass). All
flags default to stock behavior, so landing this changed nothing until a flag is set.
Every flag is **numerics-only** result-changing — the CPU `torch.randn` candidate
draws are preserved byte-for-byte in order and shape by both `skip_succeeded` and
`traj_chunk` — so they are validated by ONE bundled re-anchor run, not per-flag A/Bs.
*Conditional on Box check #1 below confirming the eval-mode checkpoint verdict:* if
the predictor were somehow pickled in train mode, `eval_every`/`skip_succeeded`/
`traj_chunk` would additionally shift the dropout CUDA RNG stream (the original
analysis at the top of this doc) — the re-anchor protocol covers either case, but the
"numerics-only" label is only literally true in the expected eval-mode case.

| Flag | Where | What | Why |
|---|---|---|---|
| `plan_eval_mode=true` | plan cfg | force `model.eval()` | expected no-op (see CORRECTION); hygiene |
| `fast_tf32=true` | plan cfg | TF32 matmuls | torch 2.3 default-off; predictor GEMMs are the dominant modeled cost |
| `fast_sdpa=true` | plan cfg | `F.scaled_dot_product_attention` in `models/vit.py` | kills the (B,16,L,L) score-tensor materialization (6.6 GB fp32 @ B=300, L=588) and its ~5 walks/layer; enabler for `traj_chunk>1`. Auto-falls back to naive attention if dropout would be active (never silently changes RNG) |
| `planner.sub_planner.eval_every=999` | plan cfg | one inner eval per `plan()` (i=0) instead of 30 | removes 290/311 evaluator calls: ~82% of env work, ~35–59 GB subprocess IPC, 1,500 batch-N wm forwards per run. The only load-bearing output of inner evals is the all-success early break, which ~never fires at SR<1 with n≥25 |
| `planner.sub_planner.skip_succeeded=true` | plan cfg | skip rollout+scoring+refit for trajs MPC already marked successful (their taken actions are zeroed by MPC anyway); their candidate draws still execute so the RNG stream is byte-identical | ~40–48% of CEM compute at final SR≈0.8 with early successes |
| `planner.sub_planner.traj_chunk=4` | plan cfg | batch 4 trajs × 300 candidates into one rollout call | collapses the sequential per-traj python loop + 1 host sync per opt step instead of per traj. Pair with `fast_sdpa` (naive attention OOMs >~1k candidates); ~4×300×588-token activations fit comfortably in 24 GB with SDPA |

Canonical fast config — a single string to append to a `plan.py --config-name
plan_pusht.yaml` or `plan_multicolor.py` command. (Only those two configs carry the
fast keys; on `plan.yaml`/`plan_point_maze.yaml`/`plan_wall.yaml` Hydra rejects these
overrides without `+` prefixes because the keys don't exist there.)

    FAST="plan_eval_mode=true fast_tf32=true fast_sdpa=true \
    planner.sub_planner.eval_every=999 planner.sub_planner.skip_succeeded=true \
    planner.sub_planner.traj_chunk=4"

### Box checks BEFORE first fast-config use (≈35 min total)

1. **Dropout premise** (30 s): `python scripts/check_ckpt_train_mode.py <ckpt>` — see the
   CORRECTION above for what each verdict means.
2. **Find the missing wall-time** (30 min): the FLOP model says a healthy 4090 finishes an
   n=30 run in ~3–7 h *even unoptimized*; observed multi-day walls imply ~6–12% GPU
   utilization. Run `scripts/profile_plan.py --opt-steps 30 --n-evals 30 --max-iter 2`
   while watching `nvidia-smi dmon` (clocks/power/util). A power-capped/shared GPU or
   CPU-thrashed host would be worth more than every code fix combined — fix the host
   first if found.

### Re-anchor protocol (ONE bundled validation for the whole set)

The fast config's numerics shifts (TF32 rounding, SDPA reduction order, batched cuBLAS
kernel selection) can flip marginal CEM elite orderings, so absolute SR may move a
little. Do NOT A/B each flag (~4 × ~100 h). Instead:

1. Land the full fast config.
2. Re-run the 5a floor (masked-L2) once at n=100 under it (~hours post-speedup).
3. Acceptance: floor within **0.80 ± 0.08** (95% binomial CI at n=100) → adopt; quote all
   subsequent numbers as fast-config numbers, never beside legacy ones. Outside the CI →
   run one legacy-config 5a at n=50 to disambiguate config-effect vs harness bug.
4. Timing: do this BEFORE establishing the multicolor oracle ceiling (the Stage-0 gate) —
   no multicolor number is locked yet, so re-anchoring now costs nothing and every
   later headline number (oracle, `g`, floors, ablations) is natively fast-config.

### Hardware policy (from the same audit)

H100 rejected: GPU math is a small fraction of observed wall; at fp32 an H100 is
nominally *slower* than a 4090 (67 vs 82.6 TFLOPS); post-bundle Amdahl caps end-to-end
gain ≤~2× at 5–9× the $/hr → 3–9× worse $/experiment. Throughput comes from sharding
independent runs across cheap spot 4090s (never shard `n_evals` within a run — shared
sequential RNG). Revisit big-VRAM GPUs only at the V-JEPA-2-AC stretch (A100 80 GB
before H100).
