# Pusher-invariant, object-factored relational goal energy — build + D-gate decision

**Status (2026-06-16): Option 1 (relational readout energy) ADOPTED. The embodiment-contamination
floor that killed latent-L2 is DISSOLVED. The residual bottleneck is the DYNAMICS (data-starved
prediction of the *moving* block), NOT the energy — milestone shifts to corpus scale; do NOT
escalate to slots.** Phase brief: "Pusher-Invariant, Object-Factored Goal Energy". Canonical
plan: `specs/PROJECT_DEFINITIVE_PLAN.md`. Predecessor finding: the latent-L2 energy had a ~4.9/7
irreducible floor because DINOv2 global attention bakes the pusher dot into every patch token
(CLAUDE.md core-model #5; `clean-scene-pivot-recon`).

## What was built
- **R** (`dino_wm/scripts/langtable/lt_readout.py`) — per-block position+identity readout =
  the Gate-1 per-patch logistic probe, productionized. A single linear head over each 384-d
  DINOv2 patch token → 9 classes (8 FIXED_8 blocks + BG); decode = soft-argmax over patches of
  each block's logit → world (x,y) + identity confidence. **Pusher-invariant by construction**:
  it reads only the 8 block classes; the white dot patch falls to BG, so the relative read is
  unaffected by the manipulator. **Geometry fix vs Gate-1:** the probe hardcoded `half_extent=0.32`;
  the real render is `0.3048` (read from the cache). Trained on cached `visual` (dot frames,
  pusher present) → `block_xy` (free GT labels). Frozen, hand-built — NOT a learned value head.
- **h** (`lt_relplan.py:rel_cost`) — closed-form graded energy `cost = ‖pos_A − pos_B‖`
  (dense, monotone → 0 at contact) `+ lam·disturb(non-target)` `+ 0.5·offtable(A,B)`.
- **g-parser** — `(start_block, target_block)` → indices. The tuple is already cached; under
  FIXED_8 the instruction degenerates to a `color shape` bigram, so the rule-based parse is trivial.
- **Wire point** — replaces `lt_g2.py:masked_dist` (the only CEM cost; call-sites 218/225/232).
  `lt_relplan.py` is a standalone planner so the dynamics trainer (`lt_g2.py`) is untouched.

## The task has NO side relation (binding finding)
Verified in `language-table` source: `PREPOSITIONS = ['to the','towards the','close to the',
'next to the']` (synonyms.py:57) are all proximity synonyms; block2block reward is symmetric
`‖pos_A − pos_B‖ < 0.05` (block2block.py:110, `TARGET_BLOCK_DISTANCE=0.05`). So **relation S is
degenerate** (always "near"), the g-parser only extracts `(A,B)`, and **the deployable energy
needs no goal latent at all** — it minimizes the decoded A–B distance. This aligns with plan I9
("use the relation itself… matching block2block's distance reward").

## D-gate results (held-out val, n=36 episodes / 444 frames; geometry-corrected)
| gate | metric | result | threshold | verdict |
|---|---|---|---|---|
| **D1 (id)** | block-patch identity acc | **0.943** | reproduce Gate-1 ~0.92 | PASS |
| D1 (id) | same-color confusion | ≤0.03 | low | PASS |
| **D1 (pos)** | soft-argmax within-0.05u (tau=0.1) | **0.965** | high | PASS |
| D1 (pos) | hard-centroid within-0.05u | 0.881 (median 0.012u) | — | (ceiling) |
| **D1 (rel)** | decoded dist(A,B) MAE vs GT | **0.026u** | « 0.05 | PASS |
| D1 (rel) | success(dist<0.05) classif acc | **0.973** | ≥0.95 | PASS |
| D1 (rel) | near/far ordering (goal closer than start) | 34/36 | — | PASS |
| Pre-flight [1] | energy monotone along true path | 0.211u→0.092u, mono-frac 0.66 | bottoms at contact | PASS* |
| Pre-flight [2] | checkpoint round-trip | max|diff| 0.0 | <1e-6 | PASS |
| Pre-flight [3] | end-to-end tiny CEM | runs; CEM<start 8/8 | pipeline connects | PASS |
| Pre-flight [4] | one CEM plan | 11.4s → n=100 ~19 min | affordable | PASS |
| **D2 (crux)** | MOVING block A pos-err PRED vs REAL | **0.144u vs 0.035u (4×)** | within ~5–10 pts | **FAIL → dynamics** |
| D2 | STATIC blocks pos-err PRED vs REAL | 0.038u vs 0.027u | within ~5–10 pts | PASS (readout robust) |
| D2 | decoded dist(A,B) MAE vs GT, PRED / REAL | 0.103u / 0.034u | — | dynamics-limited |

\* Pre-flight [1] reads 0.092u "end" because the cache subsamples at frameskip=5 — the last
*stored* frame is ~5 env-steps before contact (`goal_xy` reaches 0.055u, <0.05 in 89% = the sim
success rate). The energy tracks approach correctly; the eval target was the truncated frame.

## Decision (per brief §2 decision tree)
- **D1 PASS** → the relational readout is representationally viable on real latents; the
  embodiment-contamination problem is **solved** (R reads block positions *through* the dot:
  0.965 within-0.05 on real dot frames; static blocks 0.038u even through the dynamics).
- **D2 split** → degrades **only on the MOVING block** (the one block the dynamics must predict),
  not on static blocks. A readout distribution-shift would degrade *all* blocks; the moving-only
  4× degradation is the unambiguous **Risk #2 signature: the DYNAMICS is the bottleneck**
  (data-starved, the known G2/G3 issue). Corroborated by TF-latent error (patch-L2 6.0 ≈ copy-last
  5.5; only ~23% better on changed patches) and by CEM exploiting model error in pre-flight [3]
  (predicted block-A latent reaches B at 0.070u while the real dynamics can't, oracle-roll 0.125u —
  the CLAUDE.md core-model #2 / G4 model-error-exploitation failure).
- **ADOPT Option 1** (relational readout energy). **Do NOT escalate to slots** (D5): D2 fails for
  dynamics reasons, not readout-specific reasons — the tree's explicit "do not escalate" branch.
  **Next milestone = corpus scale** (more data → dynamics predicts the moving block → CEM's
  predicted success tracks real success), NOT a different energy.

## I8 / I7 deviation, logged with evidence (plan §1)
- **I8** ("default to latent distance; a learned value lost 0.73<0.80"): honored — R is a *hand-built
  frozen logistic readout*, not a learned value head. Plain latent distance was shown *broken by
  embodiment contamination* (~4.9/7 floor); the relational readout is the principled fix, not an
  over-engineered value. Quasimetric remains ruled out (`docs/RULED_OUT.md`).
- **I7** ("pin one canonical contact-pose latent"): the relational readout serves I7's *intent*
  (avoid a dispersed, unscoreable goal) by scoring the relation directly rather than regressing a
  pose latent. This is a mechanism substitution, recorded here per §1.

## Experiment-log entry (Appendix C)
- **date|run|commit|seed|n:** 2026-06-16 | rel-energy-D1D2 | (pending commit) | seed 0 | n=36 ep / 444 fr
- **hypothesis:** a pusher-invariant object-factored relational readout energy dissolves the
  embodiment-contamination floor; the residual bottleneck (if any) is dynamics, not the energy.
- **setup:** R trained on `/workspace/lt_cache` train (204 traj) `visual`→`block_xy`; D1 on val
  (36 traj); D2 = oracle-action rollout through `/workspace/g2/model.pth` then decode vs GT.
- **expectation (pre-registered):** D1 reproduces Gate-1 (~0.92 id, ~0.96 pos); D2 degrades on the
  moving block if the (weak) dynamics is the limiter.
- **result:** D1 PASS (id 0.943, pos<0.05 0.965, rel-succ 0.973, dist-MAE 0.026u). D2 moving-block
  PRED 0.144u vs REAL 0.035u (static 0.038u vs 0.027u) → dynamics bottleneck. Artifacts: box
  `/workspace/readout/R.pth`; logs from `lt_readout.py` + `lt_relplan.py`.
- **read-the-failures:** the degradation is moving-block-only and CEM exploits it (success 3/8 vs
  oracle-roll 0/8) → model-error exploitation, the canonical weak-WM symptom.
- **updated belief:** the energy is correct and adopted; the wall is dynamics data-starvation.
  Milestone → corpus scale (then re-anchor TF floor + n=100 D2 against a better WM). Do NOT build slots.

## Reproduce (box, dino_wm env)
```
# R (train + D1-representational on held-out val)
cd /workspace/langtable_kit && /workspace/envs/dino_wm/bin/python -u lt_readout.py \
    --cache /workspace/lt_cache --out /workspace/readout
# h + g-parser + CEM wiring: pre-flight battery + D2-preview
/workspace/envs/dino_wm/bin/python -u lt_relplan.py --cache /workspace/lt_cache \
    --model /workspace/g2/model.pth --readout /workspace/readout/R.pth
```

## Open (next milestone, gated on corpus scale — NOT this phase)
- Formal **D2 at a better WM**: after corpus scale, re-run D2; gate = moving-block PRED within
  ~5–10 pts of REAL. **D3 (LEACE)** is moot for the adopted path (R already reads real frames;
  LEACE can't fix the dynamics) — run only if reconsidering the latent-L2 fallback.
- **D4 anti-bulldozing** tuning (`lam` don't-disturb weight) + MPC replanning — meaningful only
  once the dynamics is good enough that CEM stops exploiting model error.
- **Step 6 plannability smoke** with the relational energy against the oracle ceiling, at n=100,
  once the WM clears its re-anchored TF-latent floor (I3).
