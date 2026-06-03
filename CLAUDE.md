# CLAUDE.md — Cross-Modal Text-to-Goal Bridge for Latent World-Model Planning

> Context for agentic coding. Read this fully before editing. When repo internals are unclear, **read the actual file** — `gaoyuezhou/dino_wm` has ~6 commits and sparse docs, so do not assume APIs.

## What we're building
A single module, **`g` (the "bridge")**, that maps `(z_start, text)` → a synthesized **per-patch goal latent** in a **frozen** DINO-WM latent space, so the frozen DINO-WM dynamics + CEM planner can reach a **language-specified** goal **without ever being shown a goal image**. Testbed: a modified **multi-color PushT**. Paper 1 = this bridge + a characterization of deterministic-regression behavior on a JEPA/DINO latent.

## Core mental model (internalize before coding)
1. **A "goal" is a structured per-patch latent grid (196×384 for DINO-WM), NOT a vector.** `g` synthesizes a spatial scene latent — it is *not* an embedding projection.
2. **`g` is NOT the action-conditioned (AC) dynamics predictor.** No actions, no time axis, no autoregression, no causal mask, no rollout. `g` is a single-shot `(z_start, text) → z_goal` map. The AC *training recipe* transfers (freeze encoder, train a small module on frozen features with a latent-distance loss); the AC *architecture* does not. **Do not copy the transition ViT's design into `g`.**
3. **Frozen, always:** the DINOv2 encoder, the DINO-WM transition/dynamics model, and the CEM planner. All new learning lives in `g` (+ a thin text-side projection).
4. **`g` grounds; CEM controls.** `g` finds the named color and places the T there in latent space; the existing planner moves the T to match. The planner is untouched.

## The definitive architecture of `g`
- **Form:** DiT-style **bidirectional** transformer over the 196 patch tokens. Single-shot.
- **Size:** ~**6 layers, 6 heads, width 384** (match the token dim; ~5–10M params). Bigger than the ~19M dynamics model is unnecessary.
- **Inputs:** `z_start` (196×384 patch tokens) + DINOv2/learned positional embeddings as the token sequence; text from a **frozen** text encoder (sentence-transformer e.g. MiniLM-384, or frozen CLIP text), token-level (not pooled), projected to width 384 by a small trainable MLP.
- **Conditioning:** **cross-attention** at *every* block (patches = query, text tokens = key/value). Use adaLN only for any global scalar. (Cross-attention, not adaLN, is required for selective free-form text conditioning.)
- **Output head (resolved):** `z_goal = z_start + Δ`, a **full grid** built as start-plus-learned-change (T relocated to the named target **and** erased from its origin; static patches copied). **Do NOT output object-only patches** — that causes a double-T / off-manifold goal and breaks the energy.
- **Loss:** dense **L2** to `enc(o_goal)` (use **L1** only if/when on V-JEPA-2-AC), **up-weighted on the changed patches** (`w_i = 1 + λ·1[‖enc(o_goal)_i − z_start_i‖ > τ]`, λ≈5–10).
- **Planning energy:** native CEM cost over the **full grid minus the manipulator (pusher) patches** — `g` can't know the arm's goal-time position, so don't score it.
- **Optional:** on-manifold regularization → **include** (residual framing already helps); retrieval-to-nearest-real-latent baseline → **include** (safety floor + ablation); learned distance/calibration head → **defer** to the broader-generalization claim.

## Testbed: multi-color PushT — non-negotiables
- Multiple colored T-target **outlines (decals, no physics)** at **continuously** randomized poses; block + pusher physics unchanged.
- Text names which target; **all targets visible in the same start frame** (text is strictly load-bearing).
- **Decorrelate** the named target from geometric heuristics (it must not reliably be the nearest/most-salient).
- Success = coverage/IoU of the block with the **named** target's pose.
- Headline eval = **held-out color-location combinations** (true compositional recombination; never leak a pairing across train/test).
- Always run: **instruction-agnostic floor**, **random-target (1/k) baseline**, and **swapped-text ablation** (wrong text should send the T to the wrongly-named target — an interpretable failure).

## Codebase map (`gaoyuezhou/dino_wm`, Hydra-based, conda `environment.yaml`)
- `train.py` — training entry. Stock: `python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3`. Models save to `<ckpt_base_path>/outputs/<model_name>`.
- `plan.py` — planning entry (CEM/MPC). Stock: `python plan.py --config-name plan_pusht.yaml model_name=pusht`. General: `plan.py model_name=<m> n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30`. Logs → `./plan_outputs`.
- `conf/` — Hydra configs (`train.yaml`, `plan.yaml`, `plan_pusht.yaml`, `plan_point_maze.yaml`, `plan_wall.yaml`). **We add** `env=pusht_multicolor` and `plan_pusht_multicolor.yaml`.
- `env/` — environments. **Extend the PushT env here** for multi-color (verify the exact file).
- `models/` — encoder wrapper, transition/predictor ViT, optional decoder. **`g` goes here as a NEW module** (e.g. `models/bridge.py`); do not modify the transition model.
- `planning/` — CEM/MPC planner. **Touch here** only to add the manipulator-masked energy option.
- `datasets/` — dataset loaders. **Add** a multi-color loader yielding `(z_start, instruction, z_goal, labels)`.
- `preprocessor.py` — obs preprocessing / DINOv2 encoding; reuse for latent caching.
- `metrics/` — success/coverage metrics (reuse the PushT success fn against the named target).
- `utils.py`, `custom_resolvers.py` (Hydra resolvers), `distributed_fn/` (multi-GPU helpers).

## Setup notes
- Conda env: `environment.yaml`. **MuJoCo 210** required by the repo's env setup (PushT is pymunk, but install MuJoCo anyway). PyFleX is deformable-only — skip.
- Data: download from the OSF link in the README; PushT dataset folder is **`pusht_noise`**; set `DATASET_DIR`. Checkpoints (`pusht`/`point_maze`/`wall`) also on OSF; set `ckpt_base_path` in the plan config.
- Key planning knobs: `planner=cem`, `goal_H` (goal horizon), `planner.opt_steps` (CEM iterations), `n_evals`. Training knobs: `frameskip` (action repeat), `num_hist` (history length).

## Latent / shape conventions
- DINOv2 ViT-S/14, 224×224 → **196 patch tokens × 384-d** (14×14). Use the patch tokens as the spatial grid. **Verify CLS handling and exact tensor layout** in the `models/` encoder wrapper before writing `g`.

## Hard rules / do-NOTs
- Never unfreeze the encoder, dynamics model, or CEM.
- Never give `g` actions, time, autoregression, or a causal mask.
- `g` outputs a **full grid** (`z_start + change`), never object-only patches.
- Mask the manipulator patches out of the **CEM energy**, not out of `g`'s training target. Up-weight/mask the **changed region** in `g`'s **training loss**.
- Targets must be **continuously** placed (not k fixed slots) and **decorrelated** from "nearest."
- Never leak color-location pairings across train/test.
- Match the testbed metric: **L2 for DINO-WM**.
- Read the real file when unsure; don't invent repo APIs.

## Infra / hardware
- Dev on a **single RTX 4090 (24 GB)** — sufficient for all DINO-WM-scale work. Do logic on the M4 with a few frames, then run for real on the 4090.
- **Speed = horizontal parallelism** (multiple spot 4090s sharding eval/ablations; CPU parallelism for data gen), NOT a bigger single GPU. An 80 GB A100/H100 is only for the future V-JEPA-2-AC stretch.
- Use spot instances + frequent checkpointing for batch jobs; stop idle instances.

## Workflow conventions
- Config-first via Hydra: new behavior goes through `conf/` overrides, not hardcoding.
- Set + log seeds; follow the repo's existing logging (check whether it uses wandb and match it).
- Cache encoded latents to disk; never re-encode in a hot loop.
- New code in the existing package layout (`models/`, `env/`, `datasets/`, `planning/`); keep the bridge isolated from the frozen modules.

## Current status
- **Phase 0 in progress** — multi-color env + dataset + grounding/resolution probes + dynamics reuse-vs-retrain + **oracle ceiling**. See `PHASE_0_PLAN.md`.
- **Phase 1 (build `g`) is gated** on Phase 0 exit criteria, especially **oracle SR ≥ 0.80 on held-out combos**.

## Success criteria
- Phase 0 gate: oracle (real-goal-image) SR ≥ ~0.80 on held-out color-location combos.
- `g` success: **competitive-with-oracle** — ≥ 0.75 absolute and ≥ 0.85× oracle SR — on held-out color-location combos, with the floor/baseline/swapped-text ablations behaving as predicted.

## Glossary
- **`g` / bridge** — the new module: `(z_start, text) → z_goal` (full grid via start+change). Single-shot, frozen-everything-else.
- **AC predictor** — DINO-WM's action-conditioned transition ViT. NOT the bridge.
- **Oracle** — planning toward the real goal image's encoded latent (upper bound; sees ground truth).
- **Instruction-agnostic floor** — a text-ignoring model (proves text is necessary).
- **Held-out color-location split** — the headline generalization test (novel color↔location pairings).
- **Manipulator-masked energy** — CEM cost summed over all patches except the pusher's.
