# HANDOFF — Language Table front-of-pipeline (data + render + G1)

**Status (2026-06-16):** G0 ✅ · render dot-pusher fix ✅ · G1 ✅ (re-validated on dot render) · **G2 smoke ✅ machinery + in-principle plannability** (dynamics learns block motion; full-horizon CEM partial at smoke scale → next milestone) · **Relational goal-energy phase ✅ — Option 1 ADOPTED, dynamics diagnosed as the wall** (see `dino_wm/docs/RELATIONAL_ENERGY.md`). Canonical plan: `specs/PROJECT_DEFINITIVE_PLAN.md`. Next milestone: **corpus scale** → re-anchored TF floor + n=100 D2/oracle ceiling (+likely hierarchy for the long horizon).

**Relational goal-energy (2026-06-16):** Built `R` (pusher-invariant per-block readout = productionized Gate-1 probe, geometry-fixed 0.32→0.3048), `h` (closed-form graded `‖pos_A−pos_B‖`, side-free — block2block has NO side relation), g-parser (cached `(start,target)` tuple). **D1 PASS** on real frames (id 0.943, pos<0.05 0.965, rel-succ 0.973, dist-MAE 0.026u). **D2 → DYNAMICS is the bottleneck, not the energy** (moving block PRED 0.144u vs REAL 0.035u; static 0.038u vs 0.027u — moving-only degradation = Risk #2). Embodiment-contamination floor DISSOLVED (R reads blocks through the dot). Do NOT escalate to slots; milestone = corpus scale. Scripts: `dino_wm/scripts/langtable/{lt_readout,lt_relplan}.py`; box `/workspace/readout/R.pth`.

## Environments (LT and DINO-WM are dependency-incompatible → separate envs, data flows as files)
- **`langtable`** (Mac `/Users/Tom/miniforge3/envs/langtable`; box `/workspace/envs/langtable`), py3.10. Minimal install, **no TF/JAX/reverb/tf-agents**. Reproduce: `dino_wm/scripts/langtable/setup_langtable.sh` + `langtable_minimal.patch` (patches `utils_pybullet.py` & `oracles/plot.py` gfile→`open`; `pip install imageio`). The oracle's `tf_agents` base is shimmed at runtime by `lt_compat.py` (`install_tf_agents_shim()` before importing the oracle; `GymToTFAgentsEnv` wraps the raw gym env).
- **`dino_wm`** (box `/workspace/envs/dino_wm`, torch 2.3+cu121) — DINOv2 + sklearn for the probes.
- LT repo cloned at `/Users/Tom/Active-Projects/_external/language-table` (Mac) and `/workspace/language-table` (box).

## Env facts (verified in code)
`LanguageTable(block_mode=BLOCK_8, reward_factory=block2block.BlockToBlockReward, control_frequency=10, seed)`. obs: `effector_translation(2)`, `effector_target_translation(2)`, `instruction(512 int32)`, `rgb(180,320,3)`. `env.compute_state()` → per-block `block_<name>_{translation,orientation,mask}`. **26-dim state = 8×(xy,yaw)+effector xy.** Block order = `blocks.FIXED_8_COMBINATION` (stable): red_moon, red_pentagon, blue_moon, blue_cube, green_cube, green_star, yellow_star, yellow_pentagon. Action `Box(-0.1,0.1,(2,))` 2D delta-cartesian. Source tuple after reset: `env._reward_calculator._start_block` / `._target_block`. **IK verdict:** the xArm6 is a black-box IK servo to a 2D EE setpoint (fixed z=0.145, fixed down-rotation); state/reward depend only on block poses + 2D EE ⇒ the dot is a complete abstraction.

## Render — frame-mode contract (`lt_render.py`)
Nadir, high-mount near-orthographic (true ortho blank under headless TINY), cam_z=4.0, half_extent=0.3048, backdrop auto-masked to table color (geometry-preserving). `world_to_pixel`/patch mappings parametrized.
- **`dot`** (start/rollout): arm hidden, white dot (radius = real EE contact 0.0127u) at `effector_translation`. + proprio = EE xy (corpus writer TODO for dynamics).
- **`clean`** (goal): arm hidden, no dot, α=0 (pusher-blind by construction).
- **`arm`**: DEBUG only, not in pipeline.

## Scripts (`dino_wm/scripts/langtable/`)
`lt_compat.py` (shim+env wrap) · `lt_render.py` (3-mode render) · `lt_dump_g1.py` (REG corpus: per ep → start[clean+dot], rollout dots, goal[clean] + EE + state + source tuple) · `lt_dump_contact.py` (Slice-A same-color/same-shape contact configs) · `lt_slices.py` (baseline + Slice A/B/C + displacement) · `lt_oracle_smoke.py` · `lt_dot_samples.py`/`lt_render_samples.py` (viz).

## G1 results (Gate 1-close) — 224/ViT-S, held out by episode, n as noted
| check | metric | result | threshold | verdict |
|---|---|---|---|---|
| Baseline (clean, new render) | pos<0.05u / id | **0.988 / 0.915** | (was 0.96/0.92 old render) | improved |
| **Slice A: same-color+same-shape CONTACT** | pair pos<0.05u (8 cats, n=30 ea) | **0.98–1.00**, twin-confusion ≤0.05 | ≥0.85 graceful | **PASS (crux, large margin)** |
| Slice B: dot frames, dot-patch masked | pos<0.05u / id | **0.928 / 0.879** | ≥0.90 / ≥0.88 | PASS (recovered from arm 0.57/0.70) |
| Slice B sub: contacted/nearest block | pos<0.05u (n=132) | **0.946** | ≥0.85 | PASS |
| Slice C: dot-position | detected-dot vs proj-EE | **0.4px** | ~0 | PASS |
| Slice C: systematic | per-block baseline err | uniform 0.012–0.016u | non-systematic | PASS |
| Displacement (goal-pairs, n=123) | moved-most / toward-anchor / drift | **0.97 / 1.00 / 0.0058u** | — | labels↔motion OK |

## Reproduce (box)
```
# corpus: REG (16 parallel workers) + contact
for k in $(seq 0 15); do /workspace/envs/langtable/bin/python -u .../lt_dump_g1.py --episodes 8 --seed $k --out /workspace/g1parts/part$k.npz & done; wait
/workspace/envs/langtable/bin/python .../lt_dump_contact.py --per_cat 30 --out /workspace/lt_contact.npz
# slices
/workspace/envs/dino_wm/bin/python .../lt_slices.py --reg "/workspace/g1parts/part*.npz" --contact /workspace/lt_contact.npz
```

## G2 smoke (dynamics + CEM plannability) — front-of-pipeline finish line
Scripts: `lt_dump_traj.py` (full oracle trajectories: dot-render + 2D action + EE-proprio + 26-dim state + effector-free goal) → `lt_cache.py` (encode → cached latents, frameskip=5, proprio_dim=2, action_dim=2) → `lt_g2.py` (DINO-WM-faithful dynamics: ViTPredictor over [196 visual + proprio + action] tokens, concat_dim=0, TF next-latent MSE; §8.1 pre-flight; train; latent-space CEM, full per-episode horizon, real-history seed, **dot-masked pusher-blind energy α=0**) + `lt_g2_blockcheck.py` (I3 block-TF-error on changed patches). Corpus 240 trajs (204 train/36 val), 60 epochs.

| check | result | verdict |
|---|---|---|
| pipeline end-to-end | encode→dynamics→CEM all run | ✅ |
| §8.1 single-batch overfit | loss 2.99 → **0.007** (3000 iters) | ✅ wiring sound |
| §8.1 checkpoint round-trip | save+reload+resume 1 step | ✅ |
| dynamics TF MSE | 2.14 → **0.127** (60 ep) | learns |
| **block TF-latent-error (I3, moving patches)** | model **15.3 < copy-last 19.9** (23% better) | ✅ learns block motion |
| aggregate patch-L2 | 6.06 ≈ copy-last 5.57 | static-dominated (use changed-patch metric) |
| CEM/oracle rollout reduces dist-to-goal | **6/18 · 5/18** | ⚠️ partial — modest smoke-scale dynamics + long horizon |

**Honest verdict:** the loop runs end-to-end and the latent metric is **plannable in principle** (dynamics learns motion; CEM optimizes the masked latent cost), but **robust full-horizon plannability is not yet shown at smoke scale** (240 trajs → ~23% motion capture → autoregressive error over the ~15-22-step horizon). The fix is dynamics quality = **more data + the full G2** (re-anchored TF floor, n=100 oracle ceiling) and likely **hierarchy** for the horizon — the explicit next milestone, out of this mission's scope. Repro: `lt_dump_traj` (16 workers) → `lt_cache` → `lt_g2 --epochs 60` → `lt_g2_blockcheck`.

## Forward notes (carry into later phases)
- **g goal targets must be CANONICAL configs** (instructed block at contact pose, others at start), NOT the raw oracle end-frame (incidental drift 0.0058u) — keeps g's residual-gate "most stays put" valid (plan I7). Raw-end-frame is fine ONLY for the G2 plannability smoke (reachable by construction).
- Proprio (EE xy + action delta) into dynamics: render side done (dot); wire data side in the corpus writer for G2.
- 224/ViT-S confirmed; 448/ViT-B+reg held in reserve (not needed).
