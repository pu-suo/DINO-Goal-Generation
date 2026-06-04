# Phase 0 — Oracle-Ceiling Isolation Tests (handoff briefing)

> Paste this into a fresh session to continue without prior-context bloat.
> Runs happen on a remote GPU box (you cannot run them yourself); you produce
> commands + interpret pasted results. Code edits go on the Mac → pushed to
> GitHub → user pulls on the box. **Pushing to `main` is blocked for the
> assistant — the user pushes.** Each run prints a Success rate (SR) per MPC iter.

## The one question
The multicolor PushT **oracle** (drive the block to a text-named colored target
via a *fabricated* goal) caps at **~0.5 pose-SR on easy goals**. Localize that cap
to exactly one of: **(a) the dynamics MODEL** (multi-step rollout compounds error),
**(b) the 3 distractor decals** (visual clutter), or **(c) the fabricated goal
construction** — and within (c), is it the **goal-pose** (named, decorrelated
targets) or the **un-fakeable pusher**? Then act: model → retrain; construction →
fix the oracle energy/goal (and note it self-resolves for the bridge `g`, below).

## Project context (minimal)
- Building a text→goal bridge `g` for **DINO-WM** latent planning. Phase 0 measures
  the **oracle ceiling** before building `g`. Phase-1 gate: oracle pose-SR ≥ **0.80**
  on held-out color-location combos.
- DINO-WM = **frozen** DINOv2 ViT-S/14 encoder (196 patch tokens × 384-d) + **frozen**
  ViT dynamics + CEM/MPC planner. Plan = find actions so the predicted future latent
  matches `enc(goal_image)`.
- **Success = stock POSE criterion**: block within 20 sim-px AND angle < π/9.
- Multicolor = 4 colored T-**outline** decals (visual-only); text names the target;
  success = the block reaches the **named** target's pose.

## Metrics (what to report)
Report **pose-SR** (the gate) as the headline. The multicolor runs also print
`coverage` (≥0.95 ⇔ block within ~3–5 px) — this is a **secondary diagnostic, not
the gate** (sub-patch, unreachable by patch-latent CEM; even the stock oracle
doesn't hit it). Only flag it if it diverges weirdly from pose-SR.

## Established — do NOT relitigate
1. **Pipeline healthy.** Stock PushT oracle (real goal) ≈ **0.90** pose-SR.
2. Earlier "SR 0.0" was a **metric inversion** (coverage vs pose) — fixed (`5efbb76`).
3. Stock dynamics is OOD on multicolor → retrained model `pusht_multicolor`
   (multicolor stats; eval needs `multicolor.stats_source=multicolor`). Its
   **1-step** error ≈ baseline, but **free-rollout compounds ~2× and has plateaued**;
   NOT overfitting / NOT data-quantity (train≈test) → more epochs/data won't fix it;
   a rollout-aware recipe would.
4. Fabricated-goal pusher: `alpha=0`+hidden → sparse reward → ~0.3 (frozen);
   `goal_pusher=behind`+`alpha=1` → engagement restored, **stable ~0.5** (`451196a`).
5. **`g` implication:** `g` emits a goal **latent directly** (no rendered blue dot)
   and masks the manipulator patches from the energy ⇒ the **pusher and decal
   problems are rendering-specific and largely evaporate for `g`**. If the cap is
   construction, `g` dodges it; if the cap is the **model**, `g` inherits it.

## Repo (branch `main`)
Commits already present (`git log --oneline` to confirm): `5efbb76` pose metric ·
`451196a` goal_pusher=behind · `55ce2ab` real-goal deferral · `7e20d19` isolation
harness · plus the `goal_pusher_perturbation` key on the multicolor config.
Box: `/workspace/dino_goal`, run from `.../dino_wm`. Env: `$CKPTS` (has
`outputs/<model>`), `$DATASET_DIR`. Stock=`pusht`; multicolor=`pusht_multicolor`.

## Design principle
Run the decomposition **as a ladder on the multicolor model with the 4 decals
present on every rung**, changing exactly ONE factor per step. Decals are held
constant ⇒ they never confound the model-vs-construction verdict. No stock→multicolor
cross-model inference is used for the primary decision.

**Shared knobs (every run):** `n_evals=10` (clean 0.1 SR; treat gaps **< ~0.2 as
noise**), `seed=99` (paired goals), `planner.max_iter=10` (bounded), and
`planner.sub_planner.eval_every=999` (one SR/iter, faster, logging-only — no effect
on plan quality). **Do NOT** lower `num_samples`/`opt_steps` (would unfairly weaken
the planner). `goal_source=dset` = **real** trajectory-endpoint goals (block+pusher
physically consistent); `goal_source=named_target` = the **fabricated** oracle goal.

## STEP 0 — sanity (run first, 1 run)
Confirm the pipeline is healthy in this fresh setup before trusting anything.
```bash
# B0  stock model + real goal, full metric -> expect ~0.9. If not, STOP (env/seed/model).
python plan.py --config-name plan_pusht.yaml model_name=pusht ckpt_base_path=$CKPTS \
  goal_source=dset n_evals=10 seed=99 planner.max_iter=10 planner.sub_planner.eval_every=999
```

## CORE — the ladder (3 runs, multicolor model, decals present, block-only metric)
```bash
# M-real   real dset goal + REAL pusher.  => the multicolor model's CAPABILITY ceiling.
DATASET_DIR=$DATASET_DIR python plan_multicolor.py model_name=pusht_multicolor ckpt_base_path=$CKPTS \
  multicolor.stats_source=multicolor goal_source=dset \
  n_evals=10 seed=99 objective.alpha=1 planner.max_iter=10 planner.sub_planner.eval_every=999

# M-real-behind   real dset goal + BEHIND fabricated pusher.  vs M-real => PUSHER cost (on the multicolor model).
DATASET_DIR=$DATASET_DIR python plan_multicolor.py model_name=pusht_multicolor ckpt_base_path=$CKPTS \
  multicolor.stats_source=multicolor goal_source=dset goal_pusher_perturbation=behind \
  n_evals=10 seed=99 objective.alpha=1 planner.max_iter=10 planner.sub_planner.eval_every=999

# M-oracle   named-target goal + behind pusher (EASY: bounded dist/angle).  vs M-real-behind => GOAL-POSE/construction cost. This is the ~0.5 oracle, re-verified at matched settings.
DATASET_DIR=$DATASET_DIR python plan_multicolor.py model_name=pusht_multicolor ckpt_base_path=$CKPTS \
  multicolor.stats_source=multicolor goal_source=named_target multicolor.goal_pusher=behind \
  multicolor.combo_split=all multicolor.max_goal_dist=120 multicolor.max_goal_angle=0.3 \
  n_evals=10 seed=99 objective.alpha=1 planner.max_iter=10 planner.sub_planner.eval_every=999
```
Each rung changes ONE thing: M-real → (add fake pusher) → M-real-behind → (swap real
goal for named-target goal) → M-oracle. Model + decals are constant throughout.

**Sanity on M-real difficulty:** if M-real ≈ 1.0 with tiny `block_pos_dist` at start
(the dset goals barely move the block), the real goals are trivial → bump `goal_H=10`
and rerun M-real / M-real-behind, else the ladder is uninformative.

## OPTIONAL — only to resolve a specific branch
```bash
# T1  decals-disrupt-planning check (only needed if M-real is LOW): stock model + 3 decals + real goal.
#     Static decals are identical in start & goal so they cancel in the relative energy;
#     T1 ≈ 0.9 => decals don't disrupt planning => a low M-real is the MODEL, not decals.
python plan.py --config-name plan_pusht.yaml model_name=pusht ckpt_base_path=$CKPTS \
  goal_source=dset n_evals=10 seed=99 env_with_distractors=true env_n_distractors=3 \
  planner.max_iter=10 planner.sub_planner.eval_every=999

# T2  clean-model pusher characterization (SUPPLEMENTARY, stock model — NOT a primary multicolor inference):
python plan.py --config-name plan_pusht.yaml model_name=pusht ckpt_base_path=$CKPTS goal_source=dset \
  n_evals=10 seed=99 goal_pusher_perturbation=real   pose_only_success=true objective.alpha=1 planner.max_iter=10 planner.sub_planner.eval_every=999
python plan.py --config-name plan_pusht.yaml model_name=pusht ckpt_base_path=$CKPTS goal_source=dset \
  n_evals=10 seed=99 goal_pusher_perturbation=behind pose_only_success=true objective.alpha=1 planner.max_iter=10 planner.sub_planner.eval_every=999
```
`pose_only_success=true` is required on **stock** Test-2 runs (block-only success);
the multicolor runs are block-only already. `objective.alpha=0` only with `offmap`.

## Decision tree
Let SR_real = M-real, SR_rb = M-real-behind, SR_or = M-oracle. Gaps < ~0.2 are noise
at n=10 (rerun that pair at n_evals=30 if a verdict hinges on a sub-0.2 gap).

- **B0 ≉ 0.9** → STOP, pipeline/env/seed problem; fix before interpreting.
- **SR_real ≈ 0.5 (low)** → the multicolor MODEL can't plan even real (≤-difficulty)
  goals ⇒ **MODEL-bound** → rollout-aware retraining (`num_pred>1`/scheduled sampling;
  repo locks `num_pred=1`, needs code). Optional confirm: run **T1**; T1 ≈ 0.9 ⇒
  decals are innocent ⇒ it's the model (not the decals). Most likely outcome.
- **SR_real ≈ 0.85 (high)** → model capable AND decals don't disrupt (present here) ⇒
  the cap is the **fabricated construction**. Read down the ladder:
  - SR_rb ≈ SR_real → the behind pusher is fine ⇒ **pusher not the cap**.
  - SR_rb ≪ SR_real → the fabricated pusher hurts the multicolor model ⇒ **pusher is
    load-bearing** — but it evaporates for `g` (latent output + masked patches) ⇒ fix
    via manipulator-masked energy; don't block.
  - SR_or ≪ SR_rb → the **named-target goal pose** is the hard part (decorrelated /
    rotated targets, grounding) beyond the pusher ⇒ goal-difficulty/grounding cap.
  - SR_or ≈ SR_rb ≈ SR_real (~0.85) → the oracle itself is ~0.85 at matched settings;
    the historical "~0.5" was a difficulty/setting artifact → recheck the oracle config.
- **SR_real ≈ 0.7 (intermediate)** → rerun M-real at n_evals=30; if still ~0.65–0.75,
  it's a model×construction interaction → pursue retraining AND construction fixes.

**For `g`:** model-bound ⇒ retrain the dynamics first (goal tricks won't lift the
ceiling). Construction/pusher-bound ⇒ `g` largely sidesteps it (latent goal, masked
manipulator) ⇒ adopt the manipulator-masked energy and proceed.

## Files
- `plan.py` — `PlanWorkspace`; fake-pusher perturbation in `prepare_targets` (dset
  branch, re-renders obs_g but keeps `state_g` REAL); distractor injection at the
  `gym.make` factory (guarded on `env.name=='pusht'`).
- `plan_multicolor.py` — `MultiColorPlanWorkspace`; `multicolor.goal_pusher`
  (`hide`/`behind`); `goal_source!='named_target'` → real-goal deferral.
- `env/pusht/pusht_env.py` — optional distractor T-outline rendering.
- `env/pusht/pusht_wrapper.py` — `pose_only_success` (block-only metric, stock env).
- `conf/plan_pusht.yaml`, `conf/plan_pusht_multicolor.yaml` (isolation knobs, no-op defaults).
- Memory: `~/.claude/projects/.../memory/MEMORY.md` → `phase0-status`.
