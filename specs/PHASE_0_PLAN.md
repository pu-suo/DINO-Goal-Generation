# Phase 0 — Multi-Color PushT Setup & Dynamics Reuse/Retrain Check

**Goal of Phase 0:** stand up the modified multi-color PushT environment, generate a split-aware dataset, confirm the frozen DINOv2 latent can even support the task, decide whether the shipped DINO-WM dynamics can be reused as-is or must be retrained, and — critically — **establish the oracle's end-to-end planning success**, which is the ceiling against which `g` will be measured in Phase 1.

**Do NOT build `g` in this phase.** Phase 1 is gated on the exit criteria at the bottom.

**Testbed:** DINO-WM (`gaoyuezhou/dino_wm`), frozen DINOv2 ViT-S/14 → 196×384 patch grid, ~19M ViT dynamics, CEM/MPC planner. PushT dataset folder is `pusht_noise`; shipped checkpoint is `pusht`.

---

## Compute & hardware for Phase 0
- **Single RTX 4090 (24 GB) is sufficient for all of Phase 0.** No 80 GB card needed (that's only for the future V-JEPA-2-AC stretch).
- **Data generation is CPU-bound** — pick a 4090 host with many vCPUs, or generate on a separate cheap CPU instance / the M4 in parallel. This is the step where parallelism saves real time.
- **Budget:** ~40–160 GPU-hours total (+24–48 if a dynamics retrain is triggered), ~1–2 weeks wall-clock. Cost on a 4090 ≈ $20–80.
- Do all logic/dev on the M4 with a handful of frames (CPU/MPS) to get code correct, then push real runs to the rented 4090. Use spot + checkpointing for unattended batches.

---

## 0.0 — Infra + reproduce vanilla DINO-WM (the trust gate)

**Why first:** nothing downstream is interpretable until the stock pipeline reproduces a known number on your hardware.

**Tasks**
- [ ] Clone repo, build conda env (`environment.yaml`), install MuJoCo 210 per README (PushT itself is pymunk, but the repo's env setup expects MuJoCo). Skip PyFleX (deformable only).
- [ ] Download the OSF `pusht_noise` dataset and the `pusht` checkpoint; set `DATASET_DIR` and `ckpt_base_path` in the plan config.
- [ ] Run stock PushT planning: `python plan.py --config-name plan_pusht.yaml model_name=pusht`.
- [ ] Read `plan.py`, `planning/` (CEM loop), `models/` (encoder wrapper + transition ViT), `preprocessor.py`. Write down: exact latent shape/dtype out of the encoder, how the goal latent is formed, where the planning cost is computed, and the CEM knobs (`goal_H`, `planner.opt_steps`, `n_evals`).

**Gate:** reproduce stock PushT success rate ≈ 0.90 (within run-to-run noise). If you can't, stop and fix the environment before anything else.

---

## 0.1 — Build the multi-color PushT environment

**Locate** the PushT env in `env/` (verify the exact file) and extend it. Keep the T-block + circular pusher **physics unchanged**.

**Modifications**
- [ ] Render **N colored T-target outlines** (decals) — e.g. red / yellow / green / blue. These are **visual-only: no collision, no physics, no effect on the block**. (This is what makes dynamics reuse plausible.)
- [ ] Per-episode sampler: draw N target poses `(x, y, θ)` with **continuous** randomization across the workspace, assign colors, and pick one as the **active goal**.
- [ ] **Decorrelation constraint (non-negotiable):** the active target's pose must be uniform over the workspace and statistically independent of the block's start pose. Explicitly randomize so the named target is *not* reliably the nearest/most-salient one. Add a unit check that verifies, over a sample, that "nearest target" predicts "named target" no better than chance.
- [ ] **All N targets visible in the start frame** (and throughout) so the start latent is identical across instructions for a given physical layout — this makes the text strictly load-bearing.
- [ ] Success metric: reuse the repo's existing PushT success (coverage/IoU of the T-block) but evaluated against the **named** target's pose (find it in `metrics/` or `env/`; reuse, don't reinvent).
- [ ] Instruction generator: templated (`"push the T to the {color} target"`) **plus a small paraphrase set** (synonyms, reordered phrasings). Emit per episode: instruction string, active color, active target pose, and a flag for which paraphrase template was used (hold some templates out later).
- [ ] Add a `conf/` entry `env=pusht_multicolor` exposing: `n_targets`, color palette, randomization ranges, decorrelation toggle, marker style (outline thickness/saturation).

**Gate:** env runs; renders N continuously-placed targets; success is computed vs the named target; decorrelation check passes; instruction generator emits matched (instruction, label, goal-pose) tuples.

---

## 0.2 — Dataset generation + latent caching

**Tasks**
- [ ] Collect trajectories in the multi-color env with the same policy class as `pusht_noise` (scripted/random/noise rollouts). Parallelize across CPU workers.
- [ ] Per trajectory, store: RGB frames, actions, proprio/state, active target color + pose, instruction string, paraphrase id, and a **goal frame** (render the T aligned to the named target — needed for both the oracle and `g`'s supervision later).
- [ ] Encode all frames through frozen DINOv2 ViT-S/14 and **cache latents (196×384) to disk** (follow `preprocessor.py` conventions). Never re-encode on the fly.
- [ ] Build the **split scaffold now**, even though `g` is Phase 1: define held-out **color-location combinations** (true compositional recombination — every color and every location seen individually, only the pairings held out). Generate data split-aware and record the split manifest.
- [ ] Start modest (e.g. a few thousand trajectories). Scale only if `g` later underfits — don't over-generate before you know you need it.

**Gate:** on-disk dataset with cached latents + full label fields + a frozen split manifest; a dataloader that yields `(z_start, instruction, z_goal, labels)` for the multi-color set.

---

## 0.3 — DINOv2 representation sanity check (de-risk grounding + pose resolution)

**Why:** the two least-quantified risks from the architecture review live here — (a) can text even be grounded into these patches, and (b) does the 14×14 grid resolve the T's pose (especially orientation) finely enough for the oracle to succeed. Both are cheap to test now and expensive to discover after building `g`.

**Tasks**
- [ ] **Grounding probe:** train a tiny linear probe to identify target color from the patch features at each target's location (or verify colored-target patches are linearly separable). If colors aren't distinguishable in DINOv2 patch space, grounding is impossible → make markers larger/more saturated and re-check.
- [ ] **Pose-resolution probe:** regress block `(x, y, θ)` from the 196×384 grid; inspect the **θ error** specifically. Position will likely be fine; orientation is the suspect.

**Gate (informational, sets expectations):** color is linearly decodable from patches (grounding feasible); block θ recoverable to a tolerance compatible with the success threshold. If θ is poorly resolved, expect the oracle ceiling to cap below 0.90 and adjust the success metric / note the limitation — don't blame `g` later.

---

## 0.4 — Dynamics reuse-vs-retrain check

**Tasks**
- [ ] Run the shipped `pusht` dynamics on multi-color trajectories: 1-step teacher-forced and free multi-step rollouts.
- [ ] Measure per-patch L2 prediction error **decomposed by region** using the known object poses: `block`, `pusher`, `target-marker`, `background` patches.
- [ ] Decode free rollouts (optional decoder) and eyeball: do the colored markers stay put, and does the block evolve correctly, including where the block path overlaps a marker?

**Decision rule**
- **Reuse** (skip retrain) if: block+pusher error ≈ single-target baseline (within ~10–20%) **and** marker patches are copied stably (low error, no drift) **and** no degradation near block–marker overlaps.
- **Retrain** (→ 0.6) otherwise.

**Gate:** a clear reuse/retrain decision backed by the region-decomposed error table.

---

## 0.5 — Oracle ceiling (the gate the original Step 0 was missing)

**Why:** low per-patch prediction error does **not** imply CEM can plan to the goal. "Competitive with oracle" is meaningless until you know the oracle's number, and the oracle is `g`'s ceiling.

**Tasks**
- [ ] Plan with CEM toward `z_g = enc(o_goal)` (real goal frame, T at the **named** target) over many `(init, layout, color)` combos, **including the held-out color-location combos**.
- [ ] Adapt `plan_pusht.yaml` → `plan_pusht_multicolor.yaml`; add a `goal_source` that produces the named-target goal frame. Test both the native full-grid energy and the **manipulator-masked** energy (the manipulator mask matters more for `g`, but validate it doesn't hurt the oracle).
- [ ] Report **oracle SR** (true-env success) overall and on held-out combos.

**Gate:** **oracle SR ≥ ~0.80 on held-out combos.** If below, fix env / CEM knobs / resolution *before* Phase 1 — do not build `g` against a broken ceiling.

---

## 0.6 — (Conditional) retrain dynamics on multi-color

Only if 0.4 says retrain.
- [ ] `python train.py --config-name train.yaml env=pusht_multicolor frameskip=<match repo> num_hist=<match repo>` on the multi-color dataset (~19M model, one env).
- [ ] Re-run 0.4 (region error) and 0.5 (oracle SR) to confirm.
- [ ] ~24–48 GPU-hours on a 4090; bump to an A100 for this single run only if wall-clock is painful.

---

## Phase 0 exit checklist → green-light Phase 1
- [ ] Stock PushT SR ≈ 0.90 reproduced.
- [ ] Multi-color env: continuous placement, decorrelation verified, all targets visible, success vs named target, paraphrase-capable instruction generator.
- [ ] Split-aware dataset + cached latents + frozen split manifest + dataloader.
- [ ] Grounding feasible (color decodable from patches); pose θ resolution characterized.
- [ ] Dynamics: reuse confirmed **or** retrained and confirmed (region-error table).
- [ ] **Oracle SR ≥ 0.80 on held-out color-location combos.**

When all are checked, proceed to Phase 1 (build `g`): the single-shot `(z_start, text) → z_goal = z_start + masked residual` bridge.

---

## Risks & mitigations
- **14×14 doesn't resolve T orientation → oracle caps low.** Detected in 0.3/0.5. Mitigate by adjusting the success metric to the achievable regime and documenting it; markers won't help block resolution (DINO-WM is fixed at 224 input).
- **Colored markers perturb DINOv2 features near the block → dynamics error spikes.** Detected in 0.4. Mitigate with thinner/less-saturated outlines or non-overlapping marker placement; else retrain (cheap).
- **Markers are OOD for shipped dynamics.** → retrain (0.6).
- **Data gen too slow.** → parallelize CPU workers; it's CPU-bound, not GPU-bound.
- **"Nearest-target" shortcut leaks in.** → enforce + statistically verify decorrelation in 0.1.
