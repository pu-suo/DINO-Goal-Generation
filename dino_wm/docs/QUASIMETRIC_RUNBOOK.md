# Quasimetric cost-to-go over masked DINO latents — build + runbook

> **The one question this answers:** does an *asymmetric* learned cost-to-go close
> the gap between the masked α=0 energy (FLOOR, ~0.8) and the real-pusher proprio
> ceiling (CEILING, ~1.0) on **genuine single-T PushT**, with regular planning?
> If yes, the quasimetric signal is worth carrying to multicolor + the text→goal `g`.
>
> Code is written/tested on the Mac; **all real training + eval runs on the 4090**
> (see [[dino-wm-dev-setup]] / `scripts/setup_vastai.sh`). Nothing here fabricates
> SR numbers — the GPU commands and what each gate must show are below.

## Relaunch (recommended): one command, hands-off
On a **fresh** instance (or after a Stop/Start), the whole pipeline — env → data →
cache → calibrated train → validate → 3-way eval — runs from a single idempotent
driver that saves **every** log under one dir (the box can't scroll up):

```bash
cd /workspace
git clone https://github.com/pu-suo/DINO-Goal-Generation.git dino_goal \
    || (cd dino_goal && git pull)          # get the CURRENT code first
cd dino_goal/dino_wm
tmux new-session -d -s qm && tmux switch-client -t qm   # if already inside tmux
bash scripts/relaunch_qm.sh                # ~22 GB cache; logs -> /workspace/qm_runs/<tag>_<stamp>/
```
Re-running resumes (each stage skips if its output exists). Tunables are env vars
(`N_TRAIN=8000 bash scripts/relaunch_qm.sh`, also `FORCE_TRAIN=1`, `N_EVALS`, …); see
the header of `scripts/relaunch_qm.sh`. The tail of `relaunch.log` is the decision data
(gate verdicts + FLOOR/NEW/CEILING SR). The manual stage-by-stage commands below still
work and are what the driver calls.

## Why iqe_d0 failed, and the iqe_d1 recalibration (baked into the driver)
`iqe_d0` (60k steps, 3000 trajs, `phi_offset=30 beta=0.1`, no goal cap) **failed its
gates by silent overfitting** and its NEW eval came in *below* the 0.80 floor:
- **(c) scale FAIL:** held-out adjacent-d ≈ **9** (target ~1) while train `d_trans`
  converged to **0.6** → ~15× train/val gap = the conv encoder memorised train trajs.
- **(a) monotonicity weak (0.54).** **(b) asymmetry PASS (0.76).**
- **`d_sg` inflated to ~108** for goals ≤50 steps: `beta=0.1` is too soft to saturate
  `phi` near the offset, so the spreading objective kept pushing distances up.

The recalibrated `iqe_d1` (the driver's defaults) targets both failure modes:

| lever | iqe_d0 | iqe_d1 | why |
|---|---|---|---|
| `phi_beta` | 0.1 | **0.5** | sharper saturation → stops `d_sg` inflation |
| `max_goal_offset` | none | **32** | bound goal distance so `d_sg` has a finite target |
| `phi_offset` | 30 | **25** | just under the goal cap (target cost-to-go scale) |
| train trajs | 3000 | **6000** | more data ↓ overfitting (RAM-safe after the cache fix) |
| steps | 60000 | **20000** | ~26 epochs not ~210 |
| `p_random_goal` | 0 | **0.3** | cross-traj goals (QRL spreading / generalization) |
| `weight_decay` | 0 | **1e-4** | small AdamW WD, anti-overfit belt |

Two code hardenings make this observable and feasible (both backward-compatible):
1. **`cache_qm_latents.py` pre-allocates** the latent tensor instead of `list`+`torch.cat`
   (which held ~2× in RAM and OOM'd past a few thousand trajs) — this is what lets us
   *use* more data.
2. **`train_quasimetric.py` runs a LIVE val monitor** (`--val_every`, default 1000):
   prints the train/val `d_trans` gap every N steps and saves **`qm_head_bestval.pth`**
   at the lowest val local-cost violation. iqe_d0 overfit *blind* for 5 h; now the gap is
   visible in real time and validate/plan use the best-generalizing checkpoint.

**If `iqe_d1` still fails gate (a)/(c)** after this: the next lever is the known
**moving goal-side mask** confound — for a fixed goal, the per-pair *union* mask changes
which goal patches are zeroed at each step `t` (benign for L2, not for the conv embedding).
That's a deeper train/plan-consistency change; try more data + the bestval head first.

## What this is (and is NOT)
- A **pure V\* quasimetric distance head** `d_theta(z_a, z_b)` over the frozen
  DINO-WM latent grid (masked to object-only patches). After QRL training,
  `-d_theta(z, z_goal) ≈ V*` in **model-step units**.
- It is **NOT** QRL's latent transition model `T`, Q-function, or policy, and **NOT**
  the bridge `g`. We plan with the existing frozen DINO-WM dynamics + CEM, so we only
  need the value/cost-to-go. This is the key simplification the task mandates.
- Frozen and untouched: DINOv2 encoder, DINO-WM dynamics, CEM search math, the
  manipulator mask, the success criterion, the sped-up CEM harness.

## Files
| File | Role |
|---|---|
| `third_party/torchqmet/` | vendored IQE/MRN (BSD-3, Wang & Isola 2022). `import third_party.torchqmet` |
| `models/quasimetric.py` | `QuasimetricHead` (spatial encoder `f` + projector + IQE/MRN/sym-L2), loader |
| `scripts/cache_qm_latents.py` | full-trajectory latent cache at the **model-step grid** |
| `datasets/qm_latent_dset.py` | QRL sampler: transition pairs + value pairs + union masks |
| `scripts/train_quasimetric.py` | QRL dual-optimization training loop |
| `analysis/validate_quasimetric.py` | validation gates (a) monotonicity (b) asymmetry (c) scale |
| `planning/objectives.py::create_qm_objective_fn` | the CEM energy term |
| `plan.py` (`qm:` block) / `conf/plan_pusht.yaml` | wiring + flags |
| `analysis/run_qm_eval.sh` | three-way floor/new/ceiling eval |
| `scripts/test_quasimetric.py` | **CPU** sanity tests (run on the Mac) |

## Design decisions that matter
1. **Step/frameskip consistency.** The DINO-WM dynamics advances one *model step* =
   `frameskip=5` env frames. The cache encodes each trajectory at frames
   `[0, 5, 10, …]`, so consecutive cached latents are exactly one model-step apart.
   Training uses a constant local cost **r = −1 per model-step**, so `d_theta`
   counts model-steps and matches `goal_H`. `d_theta` takes a **single frame**
   (B,196,384) — never the `num_hist` history axis (the dynamics needs history to
   predict; the value head does not).
2. **Pose-preserving encoder `f`.** A small conv stack over the (C,14,14) grid (down
   to 4×4 before flattening) — **no global mean-pool** (that destroys the T's
   pose, the known PushT bottleneck). GroupNorm + LayerNorm only (**no BatchNorm**),
   so each CEM candidate's energy is independent of the batch (unit-tested).
3. **Identical masking, train and plan.** For *any* pair we drop the **union** of the
   two pushers' patches (the existing `manipulator_energy_mask`) from **both**
   latents. So `d_theta` always sees two grids carrying the same keep-mask — in
   training and at planning time (where the union is goal-pusher ∪ real-pusher, the
   same static per-eval mask the masked-L2 energy already uses).
4. **Energy.** `E = w_l2·masked-L2 + w_qm·d_theta(mask(z_T), mask(z_goal)) (+ α·proprio)`.
   The quasimetric is the dense long-range basin (replacing the proprio shaping); the
   terminal masked-L2 is final-pose precision. `w_qm`/`w_l2` are configurable — **tune
   the ratio on the box** (`run_qm_eval.sh sweep`). With α=0 the proprio term is off.

## Verified hyperparameters (research report / QRL paper) — the defaults
IQE-maxmean 64×32 (proj_out 2048); `eps=0.25` (NOT horizon-scaled); λ init 0.01 /
λ-lr 0.01 (→0.003 if it diverges); model lr 1e-4; batch 256; Adam.
**phi** `= -softplus(OFFSET - x, beta)`, `beta=0.1`, **OFFSET = effective horizon in
model-steps** — `cache_qm_latents.py` prints the p50/p90/max model-step trajectory
length; set `--phi_offset` near p90/max (start ~30, adjust from the print).

## Run order (GPU box)
```bash
cd dino_wm && source $WS/activate.sh          # DATASET_DIR, CKPTS, env
# 0) data + stock ckpt already present (scripts/download_data.sh)

# 1) cache model-step latents (one-time; prints model-step length hist + projected GB).
#    NOTE: full pusht_noise train is ~18.7k trajs = ~71 GB f16 -> subsample with --n_rollout
#    (a few thousand trajs is ample for the small head). The script aborts if a split's
#    projected size exceeds --max_gb (default 35). train cache big, val cache tiny:
python scripts/cache_qm_latents.py --splits train --n_rollout 6000   # ~22 GB (pre-allocated)
python scripts/cache_qm_latents.py --splits val   --n_rollout 400    # ~1.5 GB
#    -> $DATASET_DIR/pusht_noise/qm_latents/{train,val}/  (note the p90/max it prints)

# 2) train the IQE head -- calibrated iqe_d1 (see the recalibration table above).
#    The LIVE [val] lines show the train/val d_trans gap; bestval head is kept separately.
python scripts/train_quasimetric.py --out $CKPTS/qm/iqe_d1 --steps 20000 \
    --phi_offset 25 --phi_beta 0.5 --max_goal_offset 32 --p_random_goal 0.3 \
    --weight_decay 1e-4 --num_workers 4
#    watch: lambda BOUNDED, d_trans -> ~1, val/train ~1x. If lambda diverges or d
#    collapses: --lambda_lr 0.003. curves -> $CKPTS/qm/iqe_d1/train_curves.png

# 3) VALIDATION GATES on held-out val (PASS (a)+(c) before CEM) -- on the BEST-VAL head
python analysis/validate_quasimetric.py --qm_ckpt $CKPTS/qm/iqe_d1/qm_head_bestval.pth \
    --cache_dir $DATASET_DIR/pusht_noise/qm_latents --split val --out qm_outputs/validate_iqe_d1
#    (a) monotonicity: decreasing-frac mean > 0.7   (b) asymmetry: rel_gap > 0.05
#    (c) scale: adjacent d ~ 1.   Do NOT proceed if (a) or (c) fail.

# 4) (optional but cheap) tune w_qm:w_l2, then the decision run
QM_CKPT=$CKPTS/qm/iqe_d1/qm_head_bestval.pth N_EVALS=10 bash analysis/run_qm_eval.sh sweep
QM_CKPT=$CKPTS/qm/iqe_d1/qm_head_bestval.pth N_EVALS=30 W_QM=1.0 W_L2=10.0 bash analysis/run_qm_eval.sh all
#    prints FLOOR / NEW / CEILING SR on the SAME genuine held-out goals. n_evals default
#    is 30 (paired across the 3 conditions; project's ">=30 for a headline"); use 10 for
#    the sweep (a cheap directional tuning scan) and 50 only for paper-grade error bars.
```

### Optional: the asymmetric-vs-symmetric ablation (the paper's core comparison)
Train the SAME head with `--head_type sym_l2` (Euclidean instead of IQE) on identical
masked latents, validate (asymmetry gate will FAIL by construction), and eval as NEW.
If sym-L2 ≈ IQE, asymmetry isn't buying anything; if IQE > sym-L2, irreversibility
matters — a clean, novel result.
```bash
python scripts/train_quasimetric.py --out $CKPTS/qm/sym_d0 --head_type sym_l2 --steps 60000 --phi_offset 30
QM_CKPT=$CKPTS/qm/sym_d0/qm_head.pth bash analysis/run_qm_eval.sh new   # compare to iqe NEW
```

## What is verified locally (Mac, CPU) vs on the box
- **Local (`python scripts/test_quasimetric.py`, all pass):** head shapes; IQE
  self-distance ≈ 0 and asymmetry; mask zeroing + (P,)↔(B,P) broadcast equivalence;
  **the full QRL loop recovers a monotone, asymmetric, unit-local-cost quasimetric on
  synthetic structured data with bounded λ**; the CEM objective is shape-correct,
  **per-sample batch-independent**, and reduces to the masked-L2 floor at `w_qm=0`.
- **Box only (needs CUDA + pusht_noise):** the cache, real training curves, the three
  validation-gate numbers on held-out trajectories, and the floor/new/ceiling SR.

## Eval scope / "easy subset" note
Single-T **stock** pusht only (no multicolor). Goals: `goal_source=dset` (real val
trajectory segments) at `goal_H=5` — the standard DINO-WM PushT eval (≈ HWM d=25),
which is the genuine distribution the 0.8 floor / 1.0 ceiling anchors were measured on.
There is **no easy-subset filter on this path**: the `max_goal_dist`/`max_goal_angle`
crutches are multicolor-sampler-only. All three conditions re-run together on the
identical seed/n_evals/budget so they are directly comparable.

## Pitfalls / stability (from the research report)
- Dual-ascent instability → lower `--lambda_lr` toward 0.003; verify softplus(λ)≥0.
- `dim_per_component` must divide `proj_out` (asserted).
- PushT contact is mildly stochastic and the DINO latent is partial → expect some
  miscalibration; the terminal masked-L2 term is the backstop, and `w_l2` can be
  raised. If IQE is too slow over the 14×14 grid, `--head_type mrn` (~2× faster).
