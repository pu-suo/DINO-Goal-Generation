# Masked-energy results + decision (STEP 4)

Stock `pusht` model, single-T, real block goal, block-only success (`pose_only_success`:
pos < 20px AND angle < π/9). `goal_source=dset seed=99 n_evals=10` (eval_seed = 99·i+1),
`planner=mpc_cem opt_steps=30 num_samples=300 max_iter=10 goal_H=5`. Same 10 goals across
all conditions. Run: `analysis/run_masked_energy_matrix.sh all`, 2026-06-05.

| cond | alpha | mask_pusher | goal_pusher | dil | SR (n=10) | role |
|---|---|---|---|---|---|---|
| R1 | 1 | off | real    | 0 | **1.0** | ceiling (proprio + real pusher) |
| R2 | 1 | off | contact | 0 | 0.6 | broken baseline (fabricated pusher, proprio on) |
| R3 | 0 | off | contact | 0 | 0.7 | drop proprio, unmasked |
| **N1** | **0** | **on** | **real** | 0 | **0.8** | **LINCHPIN: masked object-only L2, perfect goal** |
| **N2** | **0** | **on** | **contact** | 0 | **0.8** | **the real g-deployment energy** |
| N3 | 1 | on | contact | 0 | 0.6 | mask visual but keep proprio |
| N4 | 0 | off | real    | 0 | 0.8 | alpha=0 shaping control |

(A CUDA OOM hit at the very end of the sweep; every condition had already printed its
final-eval SR, which is what the summary table reports. Re-confirm at higher n — see caveats.)

## Decision: masked raw-L2 is VIABLE → do NOT build VIP

Per the pre-registered rule:
- **N1 = 0.8 ≥ 0.8** → masked object-only L2 plans with a perfect goal. VIP likely unnecessary.
- **N2 (0.8) ≈ N1 (0.8)** → **masking fully moots the goal-time pusher.** Proceed with the masked
  energy. `g`'s deployment energy (N2: alpha=0, pusher patches dropped, *fabricated* contact
  pusher) equals the perfect-goal energy (N1). **`g` does NOT need to place the goal-time pusher
  correctly** — we mask it out and lose nothing. This resolves the pusher-fakeability risk.

## Mechanism (secondary attribution of the 1.0 → 0.6 drop)

- **Proprio drag, not visual contamination.** N3 (mask visual, keep proprio, contact) = 0.6 = R2:
  masking the visual does nothing while `alpha=1` because proprio keeps chasing the fabricated
  terminal pusher. The penalty is the **proprio channel**.
- **Lever ordering:** `alpha=0` is primary (contact 0.6→0.7), the visual mask is a smaller top-up
  (0.7→0.8) that *also* makes real==contact (robustness). Visual contamination of block patches
  is minor (~0.1).
- **alpha=0 does NOT stall here.** N4 (alpha=0, real, unmasked) = 0.8. The earlier "alpha=0
  doesn't work" was the conflated multicolor setup, not the energy.

## Caveats / what this does and does not say

- **n=10 is coarse.** 8/10 has a wide binomial CI; single cells (N1 vs R1 vs R2) aren't
  statistically separable. The *pattern* (N1=N2; N3=R2; alpha=0 healthy) is the robust signal,
  not any one number. Re-confirm N1/N2/R1 at n≥30 before quoting a headline SR.
- **Masking caps the ceiling at ~0.8** (vs 1.0 with the proprio cheat). The proprio term was
  worth ~0.2 of shaping when the goal is real. For `g` (pusher unknown → alpha=0 forced) the
  operative ceiling is N2 = 0.8. The g-gate is ≥0.75 absolute and ≥0.85× oracle, so the margin
  above 0.75 is thin (~0.05). If `g` later underperforms, *better shaping* (a learned term, e.g.
  VIP, to raise the ceiling back toward 1.0) is the lever to revisit — but it is NOT needed to
  proceed. (A learned quasimetric cost-to-go was tried for this and did NOT beat the floor — see
  docs/RULED_OUT.md.)
- **This is the STOCK single-T isolation, not the multicolor held-out gate.** It de-risks the
  masked-energy design; the multicolor oracle ceiling (Phase-0 gate, ≥0.80 held-out) is a
  separate measurement on the retrained multicolor dynamics.
- **OOM hygiene for future sweeps:** run one condition per process (already the case),
  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and avoid any other GPU user.
