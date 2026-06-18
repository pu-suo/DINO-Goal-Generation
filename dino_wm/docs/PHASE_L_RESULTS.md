# Phase L — Load-bearing / Leak / Decorrelation Results

> Goal-Image-Free Relational Planning on a Frozen DINO-WM. Phase L proves (or disproves) that the
> language command is **load-bearing** rather than that the planner exploits start-state structure.
> Gate maps to plan §7B / G4: *swapped < none < correct by a clear margin; leak probe near chance on
> the decorrelated split; n ≥ 30 per condition.* All numbers sim-grounded, success = env block2block
> `‖pos_A − pos_B‖ < 0.05u` (geometry-fixed, oracle-validated ~0.90; **never loosened**).

**Provenance:** frozen stack = DINOv2 ViT-S/14 encoder + `g2_3k_roll/model.pth` dynamics +
`readout_3k/R.pth` (hard-centroid decode). Code baseline commit `5b41a77`; Phase-L diagnostic code
(`lt_diag_leak.py`, `lt_loop.py --cmd`, `lt_loop.py --mode h3chain`, `lt_envserver.py no_terminate`)
committed at end of phase. Box: vast.ai 4090 (instance 41225553). Both metrics (success + underlying
distance) reported per condition. **Anchor = `eval_A_hard.log`, n=100, seed 0 = 0.79.**

---

## GATE VERDICT — **PHASE L: PASS**

Judged against plan §7B / G4 (*swapped < none < correct by a clear margin; leak probe near chance on
the decorrelated split; n ≥ 30 per condition*):

- ✅ **Leak near chance (L1):** goal tuple unrecoverable from the start, held out by trajectory —
  worst-case margin **+0.021** over the majority baseline, *even from ground-truth positions* (the
  upper bound). Probes overfit train to 1.000 → ample capacity, nothing to learn.
- ✅ **Decorrelated-split success, no collapse (L3):** the generator draws the goal start-independently
  (verified at source + by L1), so the eval set *is* the decorrelated split → decorrelated-split
  correct-command success = the in-distribution anchor **0.79** (n=100). No collapse is possible
  because there is no start-state correlation to remove.
- ✅ **Command is load-bearing (L2):** correct **0.79** ≫ none **0.02** (margin **+0.77** ≫ 0.15), and
  the wrong command is **actively harmful on the distance metric** (final ‖A−B‖ wrong 0.247u > none
  0.213u > correct 0.102u). *Caveat reported, not rescued:* binary `wrong 0.12 > none 0.02` is a
  symmetric-task lucky-anchor artifact (success ‖A−B‖<0.05 is symmetric, so a random wrong anchor
  occasionally coincides with the goal), not mis-grounding — the distance metric gives the intended
  ordering and L1+L3 are the load-bearing evidence (§4).
- ✅ **n discipline:** L2 all conditions n=100 (same fixed seed set 0–99); leak probe full corpus
  (train 2618 / val 462); L4/swap n=30. No sharding, no metric loosening.

**Recommendation:** the load-bearing methodology holds — the language command is necessary, the goal
is not start-state-exploitable, and the headline is not inflated → **clear to proceed to Phase G**
(NOT started here — §0.3 hard boundary). The residual ceiling (~0.79) is the frozen-WM 1-step
over-prediction on long pushes (L3 failure-staring), a frozen-thesis/scope decision, **not** a Phase-L
failure.

**Honest scope of the claim:** L2 establishes *tuple-conditioning* (the planner pursues the specified
tuple and is harmed by the wrong one); full **natural-language grounding** awaits the Phase-G VLM
parser (free-form language → tuple). L1 + L3 are the load-bearing evidence and both pass.

**§0.3 hard boundaries:** none crossed — no 448/ViT-B re-encode, no unfreezing, no corpus
scaling/regeneration, no metric change, Phase G not started.

*(L4 multi-step forecast and the swap symmetry control are diagnostics/controls below; they do not
affect this gate. **L4 flags a real Phase-G risk:** 3-step end-to-end = 0.07 — flat composition of the
single-step primitive does not hold (cross-subtask interference), so Phase G needs interference-aware
composition, not just a VLM decomposer.)*

---

## Pre-flight (§2)

- **Anchor reproduction — PASS.** `lt_loop.py --mode h3 --decode hard --n 30 --seed 0` →
  **23/30 = 0.77**, exactly matching `eval_A_hard.log`'s first-30 success *count* (the n=100 anchor's
  first-30). 6 borderline episodes (dABfin≈0.05u) flip their individual flags vs the original run =
  **GPU floating-point nondeterminism** on near-threshold cases (CEM topk over CUDA reductions); the
  aggregate rate reproduces. ⇒ box is in the expected state. **Caveat carried forward:** n=100
  success rates carry ~±0.02 run-to-run noise — far below the L2 margins, immaterial to the gate.
- **Checkpoint round-trips / env handshake / single-batch overfit:** the leak probe overfits train to
  acc 1.000 (capacity guard passes — see L1); closed-loop ckpt round-trips were cleared in Phase H
  pre-flight (encoder/dyn/readout reload-identical). Env-boundary handshake live in every h3 run.
- **Seeds:** fixed `seed=0` for the loop (per-episode env seed `0+ep`); leak-probe seeds {0,1,2}.
- **Budget (written before the sweeps):** L2 none/wrong run the full 40 model-steps/ep (no early
  success) ≈ 108s/ep single; run 2-up concurrently on the (compute-saturated) 4090 → ~5–6h for the
  pair (n=100 each). L4: 2-step n=30 ≈1h + 3-step n=30 ≈1.5h. swap control n=30 ≈0.3h. L3 reuses the
  anchor. Total remaining ≈ 8h GPU — within the overnight window; correct condition reuses the
  existing n=100 anchor (no re-run).

---

## L1 — Leak probe (the load-bearing test; weighted most) — **PASS**

**Question (I12/I14):** can the goal relation be predicted from `z_start` alone — no actions, no
language — held out by trajectory? `code: lt_diag_leak.py`, corpus `lt_cache_3k` (train 2618 / val
462 trajectories, disjoint). Targets: mover A (8), anchor B (8), **unordered pair {A,B} (28)** — the
core goal since success is symmetric. Two channels: **POS = ground-truth start positions** (16 coords
+ 28 pairwise dists → the *upper bound* on any leak) and **Z = z_start latent** (mean-pooled 384).
3 probe seeds.

| target | channel | val acc | train | uniform | majority | margin vs majority |
|---|---|---|---|---|---|---|
| mover A (8) | POS (GT, upper-bnd) | 0.140 ± 0.014 | 1.000 | 0.125 | 0.119 | **+0.021** |
| mover A (8) | Z (latent) | 0.131 ± 0.010 | 1.000 | 0.125 | 0.119 | +0.012 |
| anchor B (8) | POS (GT, upper-bnd) | 0.136 ± 0.013 | 1.000 | 0.125 | 0.145 | −0.009 |
| anchor B (8) | Z (latent) | 0.125 ± 0.016 | 1.000 | 0.125 | 0.145 | −0.020 |
| **unordered pair (28)** | POS (GT, upper-bnd) | 0.044 ± 0.004 | 1.000 | 0.036 | 0.035 | +0.009 |
| **unordered pair (28)** | Z (latent) | 0.040 ± 0.006 | 1.000 | 0.036 | 0.035 | +0.005 |

- **Chance margin used:** PASS if val acc is not materially above the train-majority baseline
  (margin ≲ 0.05). **Worst-case margin across all targets × channels = +0.021** (mover-A from GT
  positions). Unordered-pair (the actual goal) sits at 0.044 vs 0.036 chance.
- **Capacity guard:** train acc = 1.000 everywhere → the probe has ample capacity and *memorizes*
  train, yet generalization stays at chance → there is genuinely **nothing to learn**, not an
  under-powered probe.
- **Why (confirmed at source):** `reward.py::_sample_objects` draws the pair
  `rng.choice(blocks, 2, replace=False)` — **uniform random, independent of positions**; the only
  position-dependence is `block2block.py::reset` rejecting pairs that *start* within
  `TARGET_BLOCK_DISTANCE+0.01 = 0.06u`. Named-pair start dist: mean 0.209u, min 0.060u, frac<0.07u =
  0.028 → the filter is weak and excludes only already-touching pairs; it never reveals *which* pair.
- **read_the_failures:** the only above-chance signal (mover-A POS +0.021) is consistent with the
  >0.06u floor (very-close pairs are slightly less likely to be named); it does not identify the pair.
- **updated_belief:** the goal is **decorrelated from the start by construction**, empirically
  verified. This simultaneously satisfies §6 (decorrelated-split verification) — the eval generator
  (env `_sample_objects`, identical to the corpus generator) produces start-independent goals, so the
  0.79 anchor is *already* the decorrelated-split number (see L3).

**Decorrelation of the eval set (seeds 0–99):** mover-A marginal = 16/14/14/14/14/11/10/7 over the 8
blocks ≈ uniform (expected 12.5) → the L2/L3 eval set shares the decorrelated generator. ✔

---

## L4 — Multi-step compounding (DIAGNOSTIC / FORECAST — not a gate)

**PRE-REGISTERED PREDICTION (recorded before running, Appendix C):**
- Single-step primitive ≈ 0.79.
- Per-subtask success ≈ 0.72–0.78 (each subtask is a fresh single-step push from the current,
  possibly more-cluttered layout).
- 2-step **end-to-end** ≈ 0.55–0.62 (naive 0.79² = 0.62; closed-loop replanning should roughly hold
  it, mild erosion from later steps disturbing earlier pairs at ~0.017u/step bulldoze).
- 3-step **end-to-end** ≈ 0.40–0.50 (naive 0.79³ = 0.49). Closed-loop should beat the naive product.
- **Implication if 3-step lands < ~0.35:** Phase-G G.2 compositional headline is at risk; cross-subtask
  recovery needs work before investing in the VLM.

**MEASURED (n=30 per chain length, disjoint pairs, `--mode h3chain`, seed 0):**

| chain | per-subtask (ever-reached) | END-TO-END (all pairs at final) | naive persubᴺ | by position |
|---|---|---|---|---|
| 2-step | 0.65 (39/60) | **0.30** (9/30) | 0.42 | s0 0.70, s1 0.60 |
| 3-step | 0.71 (64/90) | **0.07** (2/30) | 0.36 | s0 0.73, s1 0.70, s2 0.70 |

- **Prediction outcome — MISSED, and in the pessimistic direction (kept honestly):** I predicted
  end-to-end ≈ 0.55–0.62 (2-step) / 0.40–0.50 (3-step) and that *closed-loop replanning would beat the
  naive product*. **It did the opposite** — end-to-end fell **below** the naive product (0.30 < 0.42;
  0.07 ≪ 0.36).
- **Why (confirmed by staring at the episodes):** per-subtask success holds up (~0.70, flat across
  positions → the single-step primitive is fine mid-chain), but **later subtasks physically disturb
  earlier-completed pairs.** Decisive evidence: 9/30 three-step episodes had *all three* subtasks reach
  (`[Y,Y,Y]`), yet only **2** ended e2e=1 — the other **7** were `[Y,Y,Y] e2e=0`: every pair was
  momentarily satisfied, then a later push knocked an earlier one apart (blocks bump on the small table
  even with disjoint block sets). A secondary contributor: per-subtask is the optimistic "ever reached"
  metric, which doesn't persist.
- **Forecast for Phase-G G.2 (the point of L4):** flat sequential composition of the 0.79 single-step
  primitive is **not** enough — 3-step end-to-end (0.07) is far below single-step. The fix is **not**
  primarily a better VLM decomposer; it is **interference-aware composition** — order subtasks to
  minimize disturbance and/or re-verify-and-re-fix the *whole* chain (closed-loop over all pairs, not
  just within each subtask). **This is a low-regret finding: ~1 day of work flags a Phase-G risk that a
  VLM would not fix.** Diagnostic only — does NOT affect the L1–L3 gate.

---

## L2 — Swapped-command ablation — **command is load-bearing (necessary + wrong actively harmful)**

**Question:** does the planner pursue the *commanded* target, and does a wrong command actively hurt?
Three conditions, **same fixed eval set (seeds 0–99, n=100), identical config** (`--decode hard cem_H=2`).
`correct` = the anchor (`eval_A_hard.log`). `none` = flat/constant energy (CEM gets no relational
gradient). `wrong` = drive the true mover toward a **different** anchor (anchor-substitution; the
symmetric metric makes an A↔B referent swap a no-op — see swap control below). Success is **always**
scored by the env on the true (A,B). **Both metrics reported** (per §2).

| condition | success @0.05u | final ‖A−B‖ (mean) | within-0.07u | bulldoze |
|---|---|---|---|---|
| **correct** | **0.79** (79/100) | **0.102u** (driven together) | 0.80 | 0.017u |
| **none** | **0.02** (2/100) | **0.213u** (= start 0.213u → *no motion*) | 0.06 | 0.007u |
| **wrong** | **0.12** (12/100) | **0.247u** (> start → *driven apart*) | 0.23 | 0.036u |

- **Primary contrast — the command is necessary:** correct **0.79** ≫ none **0.02**, margin **+0.77**
  (≫ the 0.15 "clear" threshold). Without a relational gradient the planner takes no directed action
  (`none` final dist = start dist exactly → the block does not move; the 2 successes are near-floor
  pairs that started just above the 0.06u filter and jittered under 0.05u).
- **Wrong command is actively harmful (distance metric):** **correct 0.102u < none 0.213u < wrong
  0.247u** — the intended `wrong < none < correct` ordering holds on the continuous metric. Pursuing
  the wrong anchor *increases* ‖A−B‖ (drives the mover away) and bulldozes 5× more than `none`.
- **The one literal deviation, reported honestly:** on the *binary* metric, `wrong` 0.12 > `none` 0.02
  (not strictly below). **This is a symmetric-task artifact, pre-identified, not a grounding failure:**
  success is `‖A−B‖<0.05` (symmetric), so a randomly chosen wrong anchor that happens to sit near the
  true B drags the mover into B's neighborhood by spatial luck — the planner is faithfully executing
  the *wrong* command, which occasionally coincides with the goal. The no-motion `none` floor (0.02)
  is *structurally below* this lucky-anchor floor; a binary "wrong < none" is ill-posed here. The
  protocol's required second metric (distance) gives the correct ordering, and the **load-bearing
  evidence is L1 + L3** (per §4), both clean PASSES.

**read_the_failures (≥30 per condition):**
- `none` (98 failures): the block is essentially static — final ‖A−B‖ == start in every episode
  (mean 0.213u both); no cluster beyond "took no directed action." Confirms the null is a true null.
- `wrong` (88 failures): the mover is driven toward the substituted anchor → final ‖A−B‖ > start in
  the large majority (mean 0.247 > 0.213); the 12 lucky successes are episodes where the wrong anchor
  lay near the true B. min-dist mean 0.152u (it *does* move blocks — just toward the wrong place).
- `correct` (21 failures): distance-bound (far-pair overshoot + last-0.05u), analyzed under L3 — no
  goal-identity errors.

**Swap symmetry control (n=30, seeds 0–29):** push true-B → true-A instead of A → B. **Result: 0.67
(20/30)** ≈ correct on the same seeds (0.77; the ~0.10 gap is the mover-identity effect of pushing B
vs A + n=30 noise), and **far above none 0.02 / wrong 0.12**. This confirms the relation is symmetric:
swapping which named block is mover vs anchor still solves the task. The plan's generic "A↔B referent
swap" is therefore a *no-op* (not a wrong command) — which is exactly why **anchor-substitution** is the
load-bearing "wrong." Reported as a control, NOT part of the correct/none/wrong gate.

## L3 — Decorrelated-split success — **= the anchor (no inflation)**

**Question:** was the 0.79 anchor inflated by start-state correlation, or does it hold when the goal
is independent of the start?

**Result:** the eval generator draws the goal pair start-independently (`_sample_objects`, verified at
source + by L1 at chance even from GT positions), so the eval set used for the 0.79 anchor **is
already the decorrelated split** — there is no separate "correlated" split to compare against because
the task never correlates the goal with the start. Eval episodes are env-generated fresh trajectories
(seeds 0–99), held out from training by construction. Therefore:

- **decorrelated-split correct-command success = 0.79 (n=100, `eval_A_hard.log`)** = the in-distribution
  anchor; underlying final ‖A−B‖ mean 0.102u, dABmin 0.082u (within-0.07u = 0.80).
- There is **no collapse** because there was no start-state correlation to remove (L1). A "sharp
  collapse on a decorrelated split" is the failure mode this test screens for; it cannot occur here.

**read_the_failures (all 21 anchor failures of n=100 stared at):** the decisive observation for L3 is
that **every failure drives the mover toward the *correct* anchor** — there is no goal-identity error,
no "pushed toward the wrong block." Clusters:
- **Far-pair overshoot (~12, dominant):** d0>0.25u with dABfin > d0 (ep47 0.390→0.561, ep04
  0.323→0.434, ep81 0.380→0.522) — the frozen WM over-predicts long-push slide, the block sails past B.
- **Near-miss, last 0.05u (~5):** reached dABmin 0.06–0.10u but couldn't close (ep00 0.060, ep74 0.095,
  ep42 0.098) — precision + small overshoot.
- **No-progress (~4):** block barely moves toward B; 7/21 failures involve **red_moon** (the weakest
  readout even after hard-centroid decode). Render-drift flagged on 10/21 — a scatter *symptom*, not the
  cause.

⇒ the residual is the **frozen-dynamics 1-step over-prediction on long pushes** (+ red_moon readout),
NOT start-state exploitation or mis-grounding. This is the same ceiling the Phase-H sweep identified;
Levers C/D failed to move it. It bounds the headline at ~0.79 and is a frozen-thesis/scope decision,
not a Phase-L gate failure.

**updated_belief:** the headline 0.79 is **not** start-state exploitation; it is genuine
command-conditioned planning on a start-independent goal distribution. (Robustness note: a second
seed-block was not run — L1 already proves decorrelation, so re-measuring on more decorrelated seeds
would only re-estimate the same 0.79 ± ~0.02 GPU-noise; it is not gate-required.)

---

## Honesty framing (§4) — what each test does and does NOT prove

- **L2 = tuple-conditioning, necessary but near-tautological.** The energy is closed-form and points
  the optimizer at whatever tuple it is given, so "correct ≫ none / wrong" shows the optimizer
  *respects the specified tuple* — it is **not**, by itself, evidence of language *grounding*. The
  parser's free-form-language → tuple mapping is untested until the Phase-G VLM.
- **L1 (leak) + L3 (decorrelated success) are the load-bearing evidence:** the goal cannot be
  recovered from the start, and the headline holds on the start-independent distribution.
- **Symmetry caveat (real finding):** block2block success ‖A−B‖<0.05 is symmetric, so the plan's
  generic "A↔B referent swap" is a no-op (≈ correct) — *not* a wrong command. The load-bearing "wrong"
  here is **anchor-substitution** (drive the true mover toward a different block). The A↔B swap is run
  separately as a symmetry control.
- **Sim-scoped caveat:** observations are rendered top-down from privileged sim state; a real robot
  would need a pose estimator. Train/test consistent, no goal info leaks into z_start (L1).
