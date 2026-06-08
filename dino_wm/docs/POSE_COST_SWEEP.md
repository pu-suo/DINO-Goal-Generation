# POSE_COST_SWEEP.md — Projection-Metric Planning Cost for Masked-Actuator DINO-WM

> **What this doc is.** A self-contained working spec to implement and run the *first*
> experiment toward closing part of the 0.80→~0.97 planning gap in the frozen DINO-WM, in
> service of the text→latent-goal generator `g`. **§4 is the first thing to build and run.**
> Do not start `g` until the decision tree in §6 resolves.
>
> **§9 (Reconciliation notes) records where the code below diverges from the actual repo
> and what was actually built — read it together with the sketches.** The sketches here were
> written against a handoff description; when a sketch and the real interface disagree, the
> real interface wins.

---

## 0. The one-sentence task

Wire the **already-trained linear pose decoder** (from `pose_decode_probe.py`) into the CEM cost as a **projection metric** `‖W(ẑ_T − z_g)‖` mixed with the committed **masked-L2**, sweep the mix weight (pose-only and L2-only are the two endpoints of that one sweep), and measure SR at **n ≥ 50** on regular + rotation-heavy held-out single-T goals against the **0.80 floor** and **~0.97 ceiling**.

---

## 1. Corrected context (three things the prior writeup got wrong)

**(C1) The 0.2 gap is the proprio/pusher term, NOT latent-L2 quality.** The masked-L2 visual term is what *delivers* the 0.80 (committed deployable energy). The 0.80→~0.97 gap is the value of the **real-pusher proprio term** (R1→N4 in the masked-energy matrix), which `g` categorically cannot have. We are **not** "recovering 0.2 SR of lost latent shaping." We test whether a **better-conditioned readout of the same masked latent** recovers *some* of that gap deployably. Do not attribute "~0.2 SR" to the L2 term in code/logs/writeup.

**(C2) Do not justify mixing via "pose-only stalls on rotation (local minima)."** That potential-field pathology is about *greedy/gradient* following. **CEM scores whole action sequences against a terminal cost over `goal_H`** — it can accept plans that transiently *increase* pose-distance. The honest reasons to mix: **(a) risk management** — as the pose-term weight → 0 the cost provably recovers L2-only's 0.80, so the combined cost has a worst-case floor; and **(b) discarded-information** — a 4-number pose readout throws away task-relevant latent structure (contact geometry) that masked-L2 still sees.

**(C3) The pose-decoder cost and the "Mahalanobis / learned-W" idea are ONE family.** With a linear decoder, the pose cost *is* `‖W_probe·(ẑ−z_g)‖` in `(x,y,sinθ,cosθ)` space — a projection metric with `W` fixed to the probe's weights. A freely-learned diagonal/low-rank `W` is the same object with `W` less constrained. So this is a **single sweep over a projection spectrum**, each point mixed with masked-L2:

```
raw masked-L2   →   learned diagonal/low-rank W   →   fixed pose-projection (W = probe)
   (identity)              (subsumes & may beat it)        (zero new training; FIRST run)
```

**Why this won't reproduce the quasimetric jitter.** The dropped quasimetric failed at **bootstrapped value-fitting**. Everything here is **supervised regression** against ground-truth pose — the probe proved it works at **4.4° / 5.4px with ~3.2° trajectory jitter (no pathology)**. A fixed/learned projection is a *static readout*, not a bootstrapped value. **Do not** add a Bellman target; that is the failed family.

---

## 2. The unified design (one cost, two knobs)

```
C(ẑ_T ; z_g) = C_proj(W)  +  λ_L2 · C_L2_masked

  C_L2_masked = masked-mean ‖ M ⊙ (ẑ_T − z_g) ‖²              # committed masked-L2 (the 0.80 term)
  C_proj(W)   = w_pos · pos_scale · ‖ p̂ − p_g ‖²  +  w_ang · ang_scale · ‖ (ŝ,ĉ) − (s,c)_g ‖²
```

`p, (s,c)` are decoded by the **frozen linear pose head** applied to the **masked single-frame latent** (same masking + input normalization as probe training); `M` is the committed pusher mask (1 = keep, applied identically to rollout and goal sides). Orientation distance is **chordal** on the unit circle (differentiable, no `atan2`/wrap). **The goal pose is decoded from `z_g` inside the cost — never passed as GT** (so the path is identical when `z_g = g(z_start, text)`).

**Two knobs only:** projection `W` ∈ { none (→ L2-only) ; **probe-linear (fixed)** ; learned-diagonal ; learned-low-rank } and mix weight `λ_L2` (+ the within-`C_proj` ratio `w_pos:w_ang`).

**Endpoints (the safety net):** `w_pos=w_ang=0` → pure masked-L2 (≈0.80); `λ_L2=0, W=probe` → pose-only (old "S1"); `λ_L2` large → masked-L2-dominated (≈0.80 regardless of `C_proj`).

**Principled default weighting (tie cost to the gate).** Success = `pos < 20px AND angle < 20°`. Normalize each term to its tolerance:
```
pos_scale = (1/20px)²
ang_scale = 1 / (2·sin(10°))²   ≈ 8.29     # chordal² of a 20° error
```
Use these as the *center* of the `w_pos:w_ang` sweep.

---

## 3. Step 0 — Orient in the repo (mandatory). Done — see §9 for the reconciled interfaces.

---

## 4. THE FIRST EXPERIMENT — fixed pose-projection × masked-L2 sweep

`W = probe` requires zero new training (just load the existing decoder); it covers old S1 (pose-only) and L2-only as sweep endpoints.

### Step 1 — Persist the trained linear pose decoder
`pose_decode_probe.py` now dumps `analysis_outputs/pose_decode_probe/linear_decoder.pt` after fitting the masked linear decoder (weights + per-dim normalization + prep description; see §9). Re-run the probe once on the box to regenerate it.

### Step 2 — Implement the cost
`planning/pose_projection_cost.py::create_pose_projection_fn` returns an `objective_fn(z_obs_pred, z_obs_tgt, vis_mask=None) -> (B,)` (the real CEM seam — NOT a class with a different signature). **DEPLOYABLE:** uses only the rollout terminal latent and the goal *latent*; decodes both with the frozen probe; no GT pose, no real pusher. **SUPERVISED, NOT bootstrapped.**

**Invariants enforced:** same `vis_mask` masks both `ẑ_T` and `z_g` and feeds the decoder (one mask source); single terminal frame (`[:, -1]`); batched / latent-space / `inference_mode` (no sim/re-encode in the inner loop); goal stays a latent decoded inside the cost.

### Step 3 — Hydra config
`cost=pose_projection` and `cost=masked_l2` groups (`conf/cost/`). plan.py prefers `cfg.cost` over the legacy `cfg.objective`. Overrides: `cost.lambda_l2=`, `cost.w_pos=`, `cost.w_ang=` (and `cost.alpha=0` for the masked_l2 floor).

### Step 4 — Rotation-heavy held-out subset
Eval-side goal filtering by required rotation (`|Δθ|` between start and goal block angle, from sim state at goal construction with `goal_source=dset`). **Test-set selection only — the cost stays deployable.** `goal_filter=none|rotation_heavy`, `min_rot_deg` (default 45). Sets: **regular** (`none`) and **rotation_heavy** (`|Δθ| ≥ 45°`).

### Step 5 — Run it (reproduce 0.80 first, then sweep) — see §9 for the RECONCILED commands.

---

## 5. Deliverables

`runs/posecost_summary.csv` (`cost, lambda_l2, w_pos, w_ang, goal_set, n_evals, SR`) + the `masked_l2` reference row(s). One-paragraph readout: best SR vs **0.80** and vs **~0.97**, separately for regular and rotation-heavy. No `g` work yet.

---

## 6. Decision tree

```
Reproduce floor (5a) ~ 0.80 ?
 |- NO  -> STOP. Harness/checkpoint mismatch. Fix §7-bis hygiene, re-run 5a.
 |- YES -> run sweep (5b), take best SR over the spectrum:

   best SR >= ~0.85-0.88
     -> WIN. Prefer keeping the fix in LATENT space -> run §7 learned-W follow-up; if it
        matches/beats fixed pose-projection, adopt it (keeps g a text->latent generator).
        Then build g (G_ARCHITECTURE.md), measured vs the locked deployable oracle + floors.

   best SR ~ 0.80 (plateau, no lift)
     -> Run the PREDICTED-vs-EXECUTED diagnostic BEFORE adding any term:
        on failures compare dynamics-PREDICTED terminal pose vs sim-EXECUTED terminal pose.
          |- predicted ~ executed, pose still wrong  -> cost/SEARCH limited -> §7 S3.
          |- predicted DIVERGES from executed        -> FROZEN-DYNAMICS limited. Lock 0.80 as
               the honest frozen-WM ceiling and build g against it.

   best SR < 0.80
     -> a sweep point below the floor is fine (that's why we mix); the FLOOR is the
        lambda_L2-large / w=0 endpoint, guaranteed ~0.80. If even that dips, harness bug -> §7-bis.
```

---

## 7. Deferred / conditional (do NOT build preemptively)

**Learned-`W` follow-up (only after the first run; cheap forms only).** Same cost, `W` learned by **supervised** pose regression (probe data/targets), constrained to **diagonal** or **low-rank**. Subsumes the fixed pose-projection and can keep task-relevant dims the 4-number readout discards. **RIG caveat:** *unsupervised* Mahalanobis lost to Euclidean; your `W` is **supervised**, so it may win — treat as a hypothesis to falsify. Keep it **static** (no bootstrapping).

**S3 levers (only if the diagnostic says "search/cost").** (a) **Contact-seeking** term toward the **current** block (deployable). (b) More CEM budget / longer `goal_H` / hierarchical subgoals if horizon-driven. (c) A goal-reaching classifier as a terminal tie-breaker only.

**Multicolor Phase-0 (later, separate).** Port the winning cost to the multicolor model; establish the **multicolor oracle SR ≥ 0.80 on held-out color-location combos** — the real Stage-0 gate for `g`.

---

## 7-bis. Watch-items & hygiene

- **Pin the checkpoint.** DINO-WM's PushT number swings **0.80 ↔ 0.92** by config. The 5a sanity gate is the guard; if it doesn't land near 0.80 you're likely on a different checkpoint. Record the exact `pusht` ckpt used for the floor/ceiling and this sweep.
- **DINO-WM Table 7 is not a counterargument.** "Decoder loss hurts PushT" there is about training the **encoder** with a reconstruction loss — not using a frozen decoder as a **cost readout**.
- **Deployability filter (hard constraint).** Every cost term must be computable from **live rollout latents + the goal latent only** — no real pusher, no GT goal state. Test-set difficulty filtering uses GT only to *select episodes*, never inside the cost.
- **`g`-consistency invariant.** `g` stays a **text → latent goal** generator; the dilution fix lives entirely on the **cost side**. Do not collapse `g` into a text→pose regressor.

---

## 8. Definition of done (first experiment)

1. `pose_decode_probe.py` persists `linear_decoder.pt` (weights + normalization + prep). ✅
2. `create_pose_projection_fn` implemented, batched, latent-space, masks identical on both sides, single-frame handled; `cost=pose_projection` / `cost=masked_l2` selectable via Hydra. ✅
3. Rotation-heavy goal filter implemented (test-selection only). ✅
4. 5a reproduces ~0.80 on the pinned checkpoint. ⟵ run on the box
5. 5b sweep at `n_evals ≥ 50` on both goal sets; `posecost_summary.csv` + one-paragraph readout. ⟵ run on the box
6. Decision tree (§6) resolved.

**Then, and only then, open `G_ARCHITECTURE.md`.**

---

## 9. Reconciliation notes (actual repo interfaces — what was built)

The sketches in §2/§4 were written against a handoff; these are the **real** interfaces the
code targets. Where they differ, the code follows reality.

**(R1) The cost seam is a function with the existing objective signature — not a class.**
CEM calls `loss = self.objective_fn(i_z_obses, cur_z_obs_g, vis_mask=vis_mask)`
([planning/cem.py:146](../planning/cem.py#L146)). The objective scores the **last** predicted
frame against the goal: `z_obs_pred["visual"][:, -1]` vs `z_obs_tgt["visual"][:, -1]`, both
`(B,196,384)`; `vis_mask` is `(196,)` or `None`. So `create_pose_projection_fn(...)` returns a
closure with exactly that signature. The goal already enters as a **latent**
(`z_obs_g = wm.encode_obs(obs_g)`, [cem.py:92](../planning/cem.py#L92)) — we decode it inside.

**(R2) The decoder prep (must match the probe EXACTLY).** The probe's linear decoder is fit on
the **FULL flattened latent** `reshape(196*384=75264)` (NO pooling), with **per-dimension**
standardization over 75264 dims, an intercept on the targets, and output order
**`[x_px, y_px, cos θ, sin θ]`** (cos at index 2, sin at index 3 — `atan2(out[3], out[2])`).
`linear_decoder.pt` stores `mu (75264,)`, `sd (75264,)`, `W (75264,4)`, `ymu (4,)`; decode is
`((z_flat - mu)/sd) @ W + ymu`. (The §4 sketch's mean-pool / `W:(4,d)` / separate `b` is wrong.)

**(R3) Masking.** The same per-eval `vis_mask` (the committed pusher mask, union of goal-frame
and real recorded pusher; [plan.py:213-232](../plan.py#L213-L232)) is applied to **both** latents
before the masked-L2 term **and** before decoding — one mask source, identical on both sides.
The probe trained its decoder with `dilation=0` single-pusher masks; the plan-time union mask
(dilation default 0) zeroes slightly more patches, but the probe measured pusher-contribution
≈ +0.0°, so this is low-risk. `linear_decoder.pt` records its training dilation; the cost warns
if the plan dilation differs or if `vis_mask is None` (decoder was trained masked).

**(R4) The masked-L2 term reuses the floor's normalization** (`_masked_visual_mean`:
masked-mean over kept patches × feature dim, [objectives.py:17-28](../planning/objectives.py#L17-L28)),
so `λ_L2`-large reproduces the committed floor's ranking. No proprio term (the deployable
masked-actuator energy is visual-only — the floor is `alpha=0`).

**(R5) cwd changes under `@hydra.main`.** plan.py `chdir`s into the run dir, so a relative
`decoder_ckpt` would not resolve. The factory resolves a non-absolute `decoder_ckpt` against the
**repo root** (`Path(__file__).parents[1]`), so the default
`analysis_outputs/pose_decode_probe/linear_decoder.pt` works from any cwd.

**(R6) The real floor (SR≈0.80) command** (from docs/MASKED_ENERGY_RESULTS.md, N1/N2):
`mask_pusher=true cost.alpha=0 goal_pusher_perturbation=real pose_only_success=true`, planner =
the **default `mpc_cem`** (opt_steps=30, num_samples=300, goal_H=5), `goal_source=dset`. The §4
sketch's bare `cost=masked_l2` does NOT hit 0.80 without `mask_pusher=true cost.alpha=0`.

**(R7) Hydra wiring.** `plan_pusht.yaml` gains `- cost: masked_l2` in its defaults; plan.py uses
`cfg.cost` when present, else the legacy `cfg.objective` (other env configs unchanged). The
`objective:` block is kept for backward-compat (`build_plan_cfg_dicts`); for `plan_pusht` runs,
`cost` (= identical `create_objective_fn`) takes precedence, so stock behavior is preserved.
`goal_filter` / `min_rot_deg` are added to `plan_pusht.yaml` so they are overridable without `+`.

**(R8) Rotation filter is RNG-safe for `goal_filter=none`.** The filter lives in
`sample_traj_segment_from_dset`; when `none`, the random-draw sequence is byte-identical to the
original (so the baseline goal set is unchanged). When `rotation_heavy`, it rejects segments with
`angle_diff(state[off,4], state[off+frameskip*goal_H,4]) < min_rot_deg` (block θ = state col 4),
with an attempt cap that errors clearly if the dataset is rotation-poor.

### Reconciled run commands (the actual §5 commands)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # OOM hygiene
cd /workspace/dino_goal/dino_wm && source $WS/activate.sh

# Step 1 (once): persist the linear decoder (re-runs the probe, ~1 min)
python analysis/pose_decode_probe.py --cache_dir $DATASET_DIR/pusht_noise/qm_latents
#   -> writes analysis_outputs/pose_decode_probe/linear_decoder.pt

# 5a. Sanity gate — reproduce the floor (~0.80). If not ~0.80 -> STOP (checkpoint/harness).
python plan.py --config-name plan_pusht.yaml \
  model_name=pusht goal_source=dset seed=99 n_evals=50 goal_H=5 \
  pose_only_success=true mask_pusher=true \
  cost=masked_l2 cost.alpha=0 goal_pusher_perturbation=real goal_filter=none

# 5b. The sweep (fixed W = probe; sweep lambda_l2; endpoints = pose-only & L2-dominant).
for L2 in 0.0 0.1 1.0 10.0 1000000.0; do
 for SET in none rotation_heavy; do
  EXTRA=""; [ "$SET" = rotation_heavy ] && EXTRA="min_rot_deg=45"
  python plan.py --config-name plan_pusht.yaml \
    model_name=pusht goal_source=dset seed=99 n_evals=50 goal_H=5 \
    pose_only_success=true mask_pusher=true \
    cost=pose_projection cost.lambda_l2=$L2 cost.w_pos=1.0 cost.w_ang=1.0 \
    goal_pusher_perturbation=real goal_filter=$SET $EXTRA
 done
done
#   lambda_l2=0.0 = pose-only (old S1); lambda_l2=1e6 ~ L2-only cross-check (re-hit ~0.80).
#   Then a small w_pos:w_ang sweep ({1:1,1:2,2:1}) at the best lambda_l2.
#   SR per run is printed by the evaluator ("Success rate:") and dumped to logs.json in the run dir.
```
