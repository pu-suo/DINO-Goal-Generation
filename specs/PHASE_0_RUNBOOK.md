# Phase 0 RUNBOOK — execution order on the vast.ai 4090

This is the operational companion to `PHASE_0_PLAN.md`. It lists the exact
commands, in order, with the gate after each step. Code is generated/edited on the
Mac (`dino_wm_dev`) and **run on the rented RTX 4090** (see `../SYNC.md` for the
git push/pull loop). All commands run from the `dino_wm/` directory.

Conventions:
- `$REPO` = the cloned project repo; `cd $REPO/dino_wm` before running anything.
- `$DATASET_DIR` = where datasets live; `$CKPTS` = `ckpt_base_path` (checkpoints under `$CKPTS/outputs/<name>`).
- The local Mac dev env is `dino_wm_dev` (py3.10); the GPU box uses the upstream `dino_wm` conda env.

---

## 0.0 — Infra + reproduce vanilla DINO-WM (TRUST GATE)

```bash
# one-time box setup
bash scripts/setup_vastai.sh           # conda env `dino_wm` + mujoco210 (see script)
conda activate dino_wm

# data + checkpoints from OSF (https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28)
export DATASET_DIR=/data               # must contain pusht_noise/{train,val}
export CKPTS=/ckpts                    # checkpoints/pusht under $CKPTS/outputs/pusht
# (download pusht_noise -> $DATASET_DIR ; pusht checkpoint -> $CKPTS/outputs/pusht)

# reproduce stock PushT planning
python plan.py --config-name plan_pusht.yaml model_name=pusht ckpt_base_path=$CKPTS n_evals=50
```
**GATE:** stock PushT success rate ≈ 0.90 (within noise). If not, fix env/data/ckpt before anything else.

---

## 0.1 — Multi-color env (already built + Mac-verified; confirm on the box)

```bash
SDL_VIDEODRIVER=dummy python -m pytest tests/test_multicolor_env.py -q        # expect 8 passed
SDL_VIDEODRIVER=dummy python scripts/verify_multicolor_env.py --n 6           # montage + decorrelation report
```
**GATE:** 8/8 tests pass; `P(named == nearest) ≈ 0.25`; montage shows 4 colored decals every frame + block relocated to the named target in the goal column.

---

## 0.2 — Dataset generation + latent caching

```bash
# CPU-bound; use all vcpus. Start modest, scale later only if g underfits.
python scripts/gen_pusht_multicolor.py --out $DATASET_DIR/pusht_multicolor \
    --n_train 2000 --n_val 200 --n_test 400 --T 100 --workers $(nproc)

# encode start+goal frames through frozen DINOv2 (GPU)
python scripts/cache_latents.py --data_path $DATASET_DIR/pusht_multicolor --device cuda
```
**GATE:** `$DATASET_DIR/pusht_multicolor/{train,val,test}` with states/actions/velocities/
seq_lengths/labels + obses/*.mp4 + goal_obses/*.png; `split_manifest.json` (frozen held-out
combos); `latents/<split>/{start,goal}_latents.pth` (N,196,384). Dataloaders:
`load_pusht_multicolor_slice_train_val` and `load_multicolor_latent_goal`.

---

## 0.3 — DINOv2 representation sanity (grounding + pose θ)

```bash
python analysis/grounding_probe.py --data_path $DATASET_DIR/pusht_multicolor --split train
python analysis/pose_probe.py      --data_path $DATASET_DIR/pusht_multicolor --split train
```
**GATE (informational):** grounding `color_only_acc` >> chance (0.25) and `grounding_feasible: true`
(esp. blue vs the RoyalBlue pusher). Pose probe: note `theta_mae_deg` — if θ is poorly resolved,
expect the oracle ceiling to cap below 0.90 and adjust the success metric (don't blame `g`).
If a color is weak: bump `outline_thickness`/saturation in `conf/env/pusht_multicolor.yaml` +
`multicolor_common.DEFAULT_PALETTE`, regenerate, re-cache, re-probe.

---

## 0.4 — Dynamics reuse-vs-retrain

```bash
python analysis/dynamics_check.py model_name=pusht ckpt_base_path=$CKPTS \
    --data_path $DATASET_DIR/pusht_multicolor --split test --n_traj 50 --horizon 10 \
    --baseline_pusht_noise $DATASET_DIR/pusht_noise/val
```
**GATE:** region-decomposed error table + a printed REUSE/RETRAIN decision. REUSE if marker
patches are copied stably (target≈background error, no drift) AND block+pusher error ≈ the
single-target baseline (≤~1.2×). Otherwise RETRAIN → 0.6.

---

## 0.5 — Oracle ceiling (THE Phase-1 gate)

```bash
# headline: held-out color-location combos, native full-grid energy
python plan_multicolor.py model_name=pusht ckpt_base_path=$CKPTS n_evals=50 \
    multicolor.combo_split=heldout DATASET_DIR=$DATASET_DIR

# sanity: train combos (should be >= heldout)
python plan_multicolor.py model_name=pusht ckpt_base_path=$CKPTS n_evals=50 \
    multicolor.combo_split=train

# manipulator-masked energy (validate it doesn't hurt the oracle)
python plan_multicolor.py model_name=pusht ckpt_base_path=$CKPTS n_evals=50 \
    multicolor.combo_split=heldout multicolor.use_manipulator_mask=true \
    objective.alpha=0 multicolor.mask_tag=_masked
```
**GATE:** **oracle SR ≥ ~0.80 on held-out combos.** If below, fix env / CEM knobs (`goal_H`,
`planner.sub_planner.opt_steps`, `num_samples`) / resolution BEFORE Phase 1.

Always also run the predicted ablations (interpretability, not gates):
- instruction-agnostic floor (text ignored),
- random-target 1/k baseline,
- swapped-text (wrong color → block goes to the wrongly-named target).
(These become first-class once `g` exists; for the oracle, `combo_split=all` + shuffling the
named target documents the baselines.)

---

## 0.6 — (Conditional) retrain dynamics on multi-color  [only if 0.4 says RETRAIN]

```bash
python train.py --config-name train.yaml env=pusht_multicolor \
    frameskip=5 num_hist=3 ckpt_base_path=$CKPTS DATASET_DIR=$DATASET_DIR
# then re-run 0.4 (region error) and 0.5 (oracle SR) with model_name=<new multicolor model>
# for 0.5 with a retrained model: add multicolor.stats_source=multicolor
```

---

## Exit checklist → green-light Phase 1
- [ ] Stock PushT SR ≈ 0.90 reproduced (0.0)
- [ ] Multi-color env: continuous + decorrelated + all targets visible + named-target success (0.1)
- [ ] Split-aware dataset + cached latents + frozen split manifest + dataloaders (0.2)
- [ ] Grounding feasible; pose θ resolution characterized (0.3)
- [ ] Dynamics reuse confirmed OR retrained + confirmed (0.4 region table)
- [ ] **Oracle SR ≥ 0.80 on held-out color-location combos (0.5)**
