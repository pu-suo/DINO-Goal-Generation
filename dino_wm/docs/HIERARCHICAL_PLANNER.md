# Hierarchical Closed-Loop Planner (Phase H) + Honest Evaluation (Phase E)

> Wires the rollout-trained dynamics + relational readout energy into a **live** LanguageTable
> closed loop. Integration/evaluation engineering — no large training. Code: `scripts/langtable/{lt_ipc,lt_envserver,lt_loop}.py`. Gated; numbers are sim-grounded, n≥30.

## 1. What changed vs. before
The pre-Phase-H CEM (`lt_relplan.py`, `lt_g2.py`) was **pure open-loop imagination**: it autoregressed
the frozen dynamics from the start frame and never re-observed. That is DINO-WM's *weaker* flat-CEM
mode and is exactly where model-error exploitation hides (the planner "wins" in its own head). Phase H
builds DINO-WM's actual MPC: a short latent rollout within a reliable window, **execute K=1, re-observe
from the real sim, replan** (receding horizon). Success is the **sim's** verdict, never the WM's.

## 2. Architecture: two-process env boundary (the load-bearing decision)
The sim and the learner cannot co-exist in one conda env:
- `base` (`/opt/conda`): torch + CUDA + dinov2; **no** pybullet/cv2 → cannot run the sim.
- `langtable` (`/workspace/envs/langtable`): pybullet + language_table; **no** torch → cannot encode/plan.

So the closed loop is **two processes over a localhost socket**:
- `lt_envserver.py` (langtable env) holds the live `LanguageTable` block2block env, renders DOT frames
  via the *identical* `lt_render` path the training cache used, and ships raw RGB (224³ uint8) + ee +
  GT block_xy. Commands: `reset`, `step(env_actions)`, `close`.
- `lt_loop.py` (base env, GPU) spawns the server, encodes frames (DINOv2, exact `lt_cache` recipe,
  f16-rounded), decodes block positions with the frozen readout `R`, runs the CEM, and drives the loop.
- `lt_ipc.py`: length-prefixed pickle (stdlib only → imports in both envs).

This is standard RL/robotics infra (vector envs, Ray actors, dm_env servers): keep both proven halves
untouched, confine the new failure surface to a simple, testable wire. *Not* dependency surgery on the
torch env (which would risk silently breaking sim physics or CUDA).

### Invariants held on the boundary (verified, standing)
1. **Render-recipe consistency** — server calls the same `lt_render.render_topdown(mode='dot')`; verified
   functionally (live latents decode at the D1 ceiling) and guarded **every step** (`maxDrift` = mean
   decode-err vs GT; flagged if > `drift_thresh`=0.08u).
2. **Cross-boundary determinism** — same construction seed → identical episode through the socket
   (verified: maxdiff 0.0 at reset 0 and 2; different seed → different episode).
3. **Cost is side-free** — `block2block` reward = `‖A−B‖ < 0.05` with **no** directional term
   (`reward()` reads only the two block translations; `PREPOSITIONS` are all proximity synonyms). So the
   planning energy `‖pos_A − pos_B‖` is the correct graded surrogate of the sim's success metric.

## 3. The loop
- **Goal energy:** relational readout (`R` decodes block positions from frozen DINOv2 patches; pusher-
  invariant by construction). Sub-goals live in **readout/block-position space**, never latent-L2
  (constraint #1 — latent sub-goals would reintroduce the embodiment-contamination floor).
- **High level (geometric):** for this proximity/side-free task it reduces to a **receding "carrot"** —
  a sub-goal placed `wp_spacing`=0.10u ahead of A along A→B (or B itself when within range). Straight-line;
  obstacle-aware 2D planning only if straight-line provably fails. No learned high-level model.
- **Low level (CEM/MPC):** plan `cem_H`=2 model-steps, execute K=1 (5 env-actions), re-observe, replan.
  Cost = `‖decoded_A − subgoal‖` + contact-approach shaping + off-table + (E.1) don't-disturb.
- **Action scale (critical):** CEM samples N(0, σ=0.012) clamped to ±0.04 per env-action — **matched to
  the oracle/training action distribution** (oracle max |comp| = 0.034). Sampling larger queries the
  frozen dynamics OOD → exploitation/scatter (see §5).

## 4. Pre-flight (all PASS)
- Checkpoint round-trips (encoder/dynamics/readout reload → identical forward); no OOM (0.38 GB).
- **Env-boundary handshake** (n=30): live frame → encode → R-decode vs GT block_xy — A 0.027u/93%,
  B 0.022u/97%, decoded dist(A,B) 0.028u/90%. The D1 ceiling, now on **live** frames. corr(A-err,
  pusher→A dist) = −0.09 → R is genuinely pusher-invariant on live frames.
- One CEM plan ≈ 2.7s → an n=30 gate is minutes.

## 5. H.0b gate — single-waypoint reachability (low-level), sim-grounded, n=30
**Result: reach 0.70 (21/30); 0.81 excluding red_moon.** cem_H=2, w_approach=0.5, act_clamp=0.04,
act_sigma=0.012, seed 0. Reached via two diagnosed-then-confirmed fixes:

| stage | reach | cause / fix |
|---|---|---|
| broken | 0.17 | **action-scale OOD**: CEM σ=0.06/clamp=0.10 vs oracle max 0.034 → dynamics queried OOD → exploitation → violent shoves → scatter → R OOD → render-drift 13/30 |
| fixed actions | 0.20 | act_clamp=0.04, act_sigma=0.012 → **scatter eliminated** (drift guard green) |
| + contact-shaping | **0.70** | object-only cost is flat until contact → no CEM gradient; **contact-approach term** (pull ee to point behind A along push dir, r_contact=0.035) |

Per-block: green_cube/red_pentagon/blue_cube 1.0, green_star 0.88, yellow_*/blue_moon 0.33–0.5,
**red_moon 0/4** (weakest readout — aims wrong). Residual ceiling = **1-step model exploitation**
(realized dmin 0.055u vs WM-predicted 0.021u; present even at cem_H=1, so NOT horizon-compounding —
shortening the window, the standard remedy, provably did not help) + readout precision near 0.05u. Both
are properties of the **frozen** dynamics/readout.

## 6. H.3 — relational closed-loop success (sim-grounded block2block), n=30
**Result: SUCCESS 0.60 (18/30); 0.71 (17/24) excluding all red_moon involvement.** cem_H=2,
w_approach=0.5, wp_spacing=0.10, max_steps=50, seed 0. This is the first true plannability number for
the language→relational-goal→closed-loop-plan pipeline, and it is the SIM's own block2block verdict.

- The loop does **genuine long-horizon relational pushing**: successes at start-dist d0 = 0.42, 0.40,
  0.38, 0.36u (the carrot + K=1 MPC chains a long push); mean start-dist 0.236u, mean time-to-success
  17 model-steps.
- within-0.07u = 0.77 (the system reliably gets A near B; the final <0.05u is precision-limited).
- **red_moon is the dominant drag** — as A (can't aim: 0/... mostly fail) *and* as B (goal mislocated):
  red_moon-involved episodes 1/6 success; **red_moon-free = 17/24 = 0.71**. One weak-readout block of 8
  costs ~11 headline points. Improving R's red_moon decode is the single highest-leverage lever.
- Render-drift 2/30 (residual 1-step exploitation scatter on far pairs); a few long-distance pairs
  (0.32–0.43u) made no progress (pusher-geometry / path likely blocked → obstacle-aware waypoints would
  help those, a Phase-G-adjacent item).

## 7. E.1 — anti-bulldozing (D4)
**Baseline:** oracle (human-demo proxy) non-target displacement = **0.0056u** mean (median 0.0023,
p90 0.0152). **Raw H.3 loop (dont_disturb=0) at n=30 = 0.018u** mean (max 0.056) — small in absolute
terms (< half a block width) but ~3× oracle. The object-factored, pusher-invariant energy is
*inherently* low-bulldozing (the cost only rewards A→B and keeps the pusher behind A — no incentive to
plow neighbors). The `dont_disturb` term (penalize predicted displacement of protected blocks) is wired and tested:
**dont_disturb=0.3 (n=20) → disturb 0.018→0.013u (toward oracle) but success 11/20→10/20** on the same
seeds — the term mildly over-constrains for marginal gain, because bulldozing is already low. **Verdict:
anti-bulldozing is satisfied by construction; E.2 uses dont_disturb=0** (the cleaner, higher-success
config), with the term available for a stricter bound at ~1-episode cost.

## 7b. E.2 — n=100 anchor (the headline)
Config: cem_H=2, w_approach=0.5, dont_disturb=0, wp_spacing=0.10, max_steps=50, seed 0 (100 held-out env
configs). **SUCCESS 0.64 (64/100); 0.72 (62/86) excluding red_moon-as-A** — sim-grounded block2block.
- within-0.07u = 0.73 (vs 0.64 at 0.05u → ~9 pts are final-precision near-misses).
- start-dist mean 0.213u, final 0.119u, **median time-to-success 10 model-steps** — genuine long-horizon
  relational pushing under the frozen WM.
- bulldoze (non-target displacement) mean **0.017u** (≈3× the 0.0056u oracle; one 0.138u outlier) — small,
  no anti-bulldoze term used.
- render-drift flagged 13/100 — the residual 1-step model-exploitation scatter on far/hard pairs (the
  closed loop bounds it: 87% stay in-distribution; an open-loop planner would not).
- per-block: green_cube 0.93, blue_moon 0.90, yellow_star 0.71, green_star 0.69, yellow_pentagon 0.64,
  blue_cube 0.57, red_pentagon 0.55, **red_moon 0.14 (2/14)** — the one weak-readout block, top lever.

**Headline:** the language→relational-goal→closed-loop-MPC pipeline reaches the instructed block-to-block
relation in **64% of held-out configs (72% excluding the weak-readout block)**, sim-grounded, with frozen
encoder/dynamics/readout and no goal image — the first true plannability number for this system.

## 8. Reproduce
```
# base env on the box; ALWAYS python -u (stdout block-buffers to files)
python -u lt_loop.py --mode preflight    # pre-flight + handshake
python -u lt_loop.py --mode handshake --n 30
python -u lt_loop.py --mode h0b --n 30 --cem_H 2 --w_approach 0.5
python -u lt_loop.py --mode h3  --n 30 --max_steps 50 --cem_H 2 --w_approach 0.5 [--dont_disturb W]
# models: dyn /workspace/g2_3k_roll/model.pth (H=8 FT), readout /workspace/readout_3k/R.pth, cache /workspace/lt_cache_3k
```

## 8b. Improvement sweep (Levers B/A/C/D) — autonomous, self-gated
Baseline = E.2 anchor 0.64 (env block2block success, n=100, seed 0). Every eval = SAME fixed n=100
configs+seed. Both numbers reported where applicable; deltas labeled *measurement* vs *system*.

**Lever B — relational metric (measurement). CONCLUSION: keep 0.05u; no looser metric justified.**
0.05u is the env's OWN block2block success def (`TARGET_BLOCK_DISTANCE`), not an inherited pose
tolerance. Geometry: global closest any-two-block center distance = **0.027u** ≪ 0.05u → 0.05u is
achievable at near-contact, not too-strict. Oracle (data generator) env-success = **0.90** at 0.05u
true-final (the task's oracle ceiling; <0.95 only because the scripted RRT genuinely fails ~10%). A
looser threshold would count non-touching gaps the env calls failures = inflation. → **headline stays
0.64 @ 0.05u**; within-0.07u (0.73) is a near-miss diagnostic. Our 0.64 = **0.71× the 0.90 oracle ceiling.**

**Lever A — red_moon readout. NOT resolution-bound; a cheap DECODE fix.**
Diagnostic: over-capacity MLP made red_moon *worse* (0.18u vs linear 0.105u), BUT the binary probe
red_moon-vs-red_pentagon separates at **0.98** on frozen 224 features (all same-color pairs ~0.98). So
the features carry the distinction — **no 448 re-encode needed** (§3 boundary NOT hit). The deficit is
the soft-argmax leaking red_moon's logit onto red_pentagon's same-color patches. Fix = **hard-centroid
decode** (assign each patch to its top class first): red_moon **0.105u→0.023u** (parity), 0.5% miss, the
other 7 unregressed. → applied as `--decode hard`; n=100 eval RUNNING.

**Lever C — obstacle-aware waypoints. JUSTIFIED by C.1.** Obstruction predicts failure: at clr<0.05,
obstructed episodes succeed 0.56 vs 0.81 for clear ones (failures mean clearance 0.031u vs 0.046u;
mean d0 0.265 vs 0.184u; seed↔episode alignment 100/100). Implemented a potential-field sideways-detour
carrot in readout/block-position space (`--waypoints obstacle`). Eval pending (after +A).

**Lever D — conservative near-goal shaping.** Implemented (`--conservative` scales action clamp/sigma
when ‖A-B‖ < conservative_dist) to curb 1-step over-prediction overshoot. One bounded attempt pending.

### Ablation table (sim-grounded block2block, n=100, seed 0)  — FILLED AS EVALS COMPLETE
| config | success @0.05u (headline) | within-0.07u (diag) | red_moon | notes |
|---|---|---|---|---|
| baseline (soft, straight) | 0.64 | 0.73 | 0.14 (2/14) | E.2 anchor; 0.71× the 0.90 oracle |
| **+A (hard decode)** | **0.79** | 0.80 | **0.64 (9/14)** | **+0.15 SYSTEM**; red_moon fixed, no regression; drift 12/100 |
| +A+C (obstacle waypoints) | 0.61 | 0.63 | 0.50 | **−0.18 → C REVERTED** (negative result); drift 12→23 |
| +A+D (conservative shaping) | 0.72 | 0.76 | 0.79 | **−0.07 → D REVERTED** (negative); helped red_moon, hurt others |
| **FINAL (= +A, hard decode)** | **0.79** | 0.80 | 0.64 | **best; +0.15 over baseline, all from Lever A** |

**Lever D verdict: REVERTED (negative result).** Conservative near-goal action damping (scale clamp/sigma
×0.5 when ‖A-B‖<0.08u) slowed the final approach → episodes time out short of 0.05u (within-0.07 0.76 but
success 0.72). It helped red_moon (0.64→0.79) but hurt others (e.g. yellow_pentagon 0.79→0.50); net −0.07.
Lever A already resolved the near-goal near-misses D targeted, so D had little to add. One bounded
attempt, reverted per the keep-iff->0.79 rule; no variant-chasing.

**Lever C verdict: REVERTED (negative result).** Explicit obstacle-detour waypoints regressed −18 pts
and raised drift 12→23. The straight carrot + closed-loop MPC already navigates obstruction implicitly;
forcing sideways detours over-corrects and induces erratic pushing/scatter. C.1's obstruction↔failure
correlation reflects obstruction co-occurring with distance+exploitation, NOT blocked-path that a detour
fixes. Per the contract, no C-variant chasing. Kept config = **+A (0.79)**.

### Residual analysis (final config +A, 21/100 failures; ≥30 stared across levers)
Baseline 36 failures → +A 21. After A: red_moon-involved 17→7, near-misses 9→1 (A resolved them).
The +A residual is **obstruction 15/21, no-progress 17/21, scatter(drift>0.08) 10/21, far(d0>0.25) 12/21**
— heavily overlapping (far pairs are obstructed AND scatter). Distance is the spine: d0<0.15→0.87,
0.15–0.25→0.85, >0.25→0.67.

**Honest remaining floor (what cheap levers can't touch):** the residual is the frozen WM's **1-step
over-prediction on long pushes** (scatter) + long-horizon control on far pairs. Lever C (explicit detours)
and Lever D (near-goal damping) both *failed* to move it (and regressed) — confirming it is not a
waypoint-geometry or near-goal-overshoot problem but the frozen-dynamics contact/slide fidelity over long
pushes. The scripted oracle itself only reaches 0.90, and on far pairs the gap to it is the WM-vs-true
dynamics error compounded over the longer push. Closing it would need a better dynamics (out of scope:
frozen thesis) or a fundamentally different long-push strategy — not a cheap lever. **0.79 = 0.88× the
0.90 oracle ceiling** is the earned number with the frozen world model.

## 9. Forward (post-sweep, for separate scoping)
- **Phase L** — load-bearing/leak/decorrelation protocol (the moat), run on the closed-loop system.
- **Phase G** — generalization (harder LT variant / second domain / compositional).
- **Cheapest further headline lever is now exhausted at the readout/planner level** (A captured it). Beyond
  0.79, the lever is dynamics contact-fidelity on long pushes — a frozen-thesis / scope decision, not cheap.
