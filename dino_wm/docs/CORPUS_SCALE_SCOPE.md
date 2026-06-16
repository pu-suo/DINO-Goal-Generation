# Corpus-scale dynamics retrain — staged scope + pre-declared kill criterion

**Status (2026-06-16): SCOPED, awaiting green-light for Stage 1.** The relational energy is
adopted and proven (`docs/RELATIONAL_ENERGY.md`); D2 isolated the wall to the DYNAMICS (moving
block PRED 0.144u vs REAL 0.035u). This retrain is gated, staged, and killed on a **pre-declared
moving-patch error**, not val loss or CEM "success" (which stays untrustworthy until the WM stops
being exploitable). All numbers below are MEASURED on the box (RTX 4090, 32 CPU), not estimated.

## The honest caveat
"More data fixes the moving-block prediction" is a **hypothesis**, well-evidenced by the D2 split
but not yet proven. Stage 1 converts it into a measured data→quality curve *before* the full spend.
If the curve is flat, the lever is horizon/hierarchy or a better-conditioned WM, not data — and
we learn that for the price of a ~3 hr run, not a full corpus.

## Measured unit costs
| thing | measurement | source |
|---|---|---|
| data-gen | **~18 s/traj** single-worker (6/8 success) | timed `lt_dump_traj.py --episodes 8` |
| data-gen throughput | ~3,200 traj/hr @16 workers; ~4,800 @24 (32 CPU) | 18s/16, /24 |
| latent storage | **147 KB/frame** (196×384×2 fp16) = PLAN §9 confirmed | by construction |
| cache/traj | **~3.7 MB** (Smax=25 steps + goal, padded) | 26×147KB |
| train | **8.7 steps/s**, ~7.7 s/epoch @ current 204 traj | timed Dyn fwd+bwd, batch 32 |
| disk free | **117 GB** (/workspace, 23% used) | `df -h` |

## Staged plan (scale ONE variable: corpus size; everything else frozen)
**Stage 1 — intermediate, ~3,000 trajectories** (12.5× current 240):
- gen ~1 hr (16w) → cache ~11 GB, ~15 min encode → train ~40–60 epochs ≈ ~1.5–2 hr.
- **Wall-clock ~3–4 hr. Storage ~11 GB.** Read the moving-block PRED number → go/no-go.

**Stage 2 — full, size SET BY the Stage-1 curve** (only if Stage 1 is GREEN):
- e.g. ~10k traj: gen ~3 hr, cache ~37 GB, train ~3–5 hr → **~half a day, ~37 GB** (fits 117 GB).
- Then formal **D2 at n=100** + **Step-6 plannability smoke** vs the oracle ceiling, against a WM
  that clears its re-anchored TF-latent floor (I3).

## Pre-declared kill criterion (decide BEFORE the run)
**Primary metric:** moving-block PRED position error — R decoded on the dynamics-PREDICTED final
latent (oracle rollout) vs GT, held-out, n≥30. Tool: `lt_relplan.py --d2only` (fast checkpoint probe).
**Secondary:** changed-patch (moving) TF-L2 (`lt_g2_blockcheck.py`) — the real WM-quality signal,
NOT aggregate patch-L2 (which is static-dominated: 6.0 ≈ copy-last 5.5 hides the moving region).

| anchor | value |
|---|---|
| current (240 traj) moving-block PRED | **0.144u** |
| REAL-frame ceiling (R on real latent) | 0.035u |
| static-block PRED (already fine) | 0.038u |
| **D2 PASS (eventual)** | moving-PRED **≤ 0.05u** (success radius), target ~0.04u; ≈ predicted-moving within-0.05 ≥ ~0.90 (within ~5–10 pts of REAL ~0.97) |

**Stage-1 go/no-go @ 3k:**
- **GREEN** (data IS the lever): moving-PRED **≤ ~0.08u** (cleared >45% of the 0.144→0.04 gap) → size & launch full corpus from the curve.
- **RED** (data is NOT the lever): moving-PRED **> ~0.12u** (barely moved) → STOP corpus path; redirect to horizon/hierarchy (PLAN §5.5) or a bigger/better-conditioned dynamics model.
- **AMBER** (between): run one more point (~6k) to resolve the slope before committing.

**Kill/extend mid-run, on evidence:** at Stage-1 launch add `--ckpt_every K` to `lt_g2.py` (a few
lines), then run `lt_relplan.py --d2only --model <latest_ckpt>` every few epochs. Kill early if the
curve plateaus above RED; extend if still descending at the epoch budget. No fire-and-forget.

## §6 data-generation safeguards (the trap)
Current gen is healthy at 204 traj — **54/56 (A,B) pairs, direction +x 0.45 / +y 0.49, 13% same-color,
goal A–B 0.06u** (no PushT-style narrowness). Preserve it:
- Scale via **more seeds / more episodes-per-worker**; do NOT touch the env reset randomization or
  the reward's start/target sampling (that is what produces the diversity).
- **Bake a balance assertion into the cache step**: pair coverage, direction balance, same-color
  fraction, start-separation ≥ 0.06 — fail loudly if a bigger run drifts narrow.
- **Freeze the render contract**: dot mode (EE_RADIUS 0.0127, white dot), clean effector-free goal,
  mask_border; assert `half_extent=0.3048 / center=(0.375,0) / size=224` unchanged across the run.
  Keep the raw-oracle-end-state goal (0.006u drift) as the reachable goal. A silent renderer/oracle
  change at scale is the report-vs-reality gap that has bitten this project before.
- One config + seed-list logged; the corpus is regenerable from (seeds, episodes/worker).

## Horizon reconciliation (brief vs plan)
Not a conflict: cache `seq_lengths ≤ 25` MODEL-steps (frameskip 5) → CEM horizon ~15–22 model-steps
(the brief's number) = ~50–150 ENV-steps (the plan's I4 number). At ~22 model-steps the flat CEM
horizon is fine; hierarchy (§5.5) is the RED-branch fallback, not needed for Stage 1.

## Commands (Stage 1)
```
# gen 3k traj = 16 workers x ~190 episodes (seeds 0..15), langtable env
for k in $(seq 0 15); do /workspace/envs/langtable/bin/python -u lt_dump_traj.py \
    --episodes 190 --seed $k --out_dir /workspace/lt_traj_3k & done; wait
# cache (dino_wm env) -> train/val split
/workspace/envs/dino_wm/bin/python lt_cache.py --traj_dir /workspace/lt_traj_3k --out /workspace/lt_cache_3k
# train with periodic checkpoints (add --ckpt_every) ; probe mid-run
/workspace/envs/dino_wm/bin/python lt_g2.py --cache /workspace/lt_cache_3k --out /workspace/g2_3k --epochs 60 --ckpt_every 10
/workspace/envs/dino_wm/bin/python lt_readout.py --cache /workspace/lt_cache_3k --out /workspace/readout_3k
/workspace/envs/dino_wm/bin/python lt_relplan.py --d2only --cache /workspace/lt_cache_3k \
    --model /workspace/g2_3k/model.pth --readout /workspace/readout_3k/R.pth
```
(`--ckpt_every` is the only code change needed; everything else is committed.)
