# Masked-energy inspection note (STEP 1)

Goal of the experiment: on the **stock** PushT model, decide whether a g-style energy
(visual latent L2 over **object** patches only, pusher patches dropped, proprio term OFF)
plans to **SR ≥ 0.8 with a perfect (real) goal**. If yes → masked raw-L2 is viable, no VIP.
If it stalls even with a perfect goal → the masked object-only energy is too flat → shaping
problem → justifies a learned shaping term (e.g. VIP). (A learned quasimetric cost-to-go was
tried for this and ruled out — see docs/RULED_OUT.md.)

This note answers (a)–(e) from the actual code before any code change.

---

## (a) Where the planning cost is computed, and how the goal latent enters

- **Cost fn:** [planning/objectives.py:30-45](../planning/objectives.py#L30-L45) `objective_fn_last`
  (mode=`last`, the configured mode). Per-eval scalar over a batch of CEM samples.
- **Goal latent enters** at [planning/cem.py:80](../planning/cem.py#L80):
  `z_obs_g = self.wm.encode_obs(trans_obs_g)` — `obs_g` is encoded ONCE per plan, then
  broadcast to all 300 samples ([cem.py:96-101](../planning/cem.py#L96-L101)) and passed as
  `cur_z_obs_g` to the objective at [cem.py:117](../planning/cem.py#L117).
- `encode_obs` ([models/visual_world_model.py:120-134](../models/visual_world_model.py#L120-L134))
  returns `{"visual": (B,T,P,D), "proprio": (B,T,proprio_emb)}` with **P=196, D=384**.
- The rollout's predicted latents `i_z_obses` come from `self.wm.rollout(...)`
  ([cem.py:111](../planning/cem.py#L111)); the objective scores the **last** predicted frame
  `z_obs_pred["visual"][:, -1:]` against `z_obs_tgt["visual"]`.

**Conclusion:** the only place to change cost is `objective_fn` + whatever feeds it a mask.
Neither the encoder, the predictor (`wm.rollout`), nor the CEM search loop need to change.

## (b) Current energy decomposition + where alpha lives + what proprio scores against

[objectives.py:39-44](../planning/objectives.py#L39-L44):

```
loss_visual  = MSE(z_pred.visual[:, -1:], z_goal.visual)   # mean over (T, P, D)
loss_proprio = MSE(z_pred.proprio[:, -1:], z_goal.proprio)  # mean over (T, proprio_emb)
loss         = loss_visual + alpha * loss_proprio
```

- **`alpha`** is bound at fn-creation: `create_objective_fn(alpha, base, mode)`, called via
  Hydra from `cfg.objective.alpha` ([plan.py:143-145](../plan.py#L143-L145)). `alpha=0` drops
  the proprio term **entirely** — confirmed, it multiplies the whole `loss_proprio`.
- **`loss_proprio` is scored against `z_goal.proprio`** = the encoded proprio of `obs_g`. In
  PushT, proprio = the **pusher (agent) xy** of the goal frame. So with `alpha>0` the planner
  is pulled toward the **goal-frame pusher pose**. When the goal pusher is fabricated (contact),
  `alpha>0` makes proprio chase a fabricated terminal pusher. `alpha=0` removes that drag.

## (c) Image→patch geometry and patch ordering

- Encoder = DINOv2 ViT-S/14 on a 224×224 frame → 16×16 = 256? **No** — repo uses a
  **14×14 = 196** patch grid (P=196 in `encode_obs`). Patch side = 224/14 = 16 px in render
  space; equivalently a 14-px patch in a 196-px virtual frame. The existing
  `pusher_patch_mask` ([env/pusht/multicolor_common.py:119-135](../env/pusht/multicolor_common.py#L119-L135))
  works in that virtual frame and is **scale-invariant** (agent xy and patch centers both scale
  with the virtual image size, so the `dist² ≤ r²` test is unaffected by 196-vs-224).
- **Sim→patch map:** `ax = agent_x · grid / sim`, patch (row `ri`, col `ci`) center at
  `((ci+0.5), (ri+0.5))` in patch units; drop patch if within `r` of `(ax, ay)`, where the
  pusher radius is `15 sim-px` + a `pad·patch` cushion (~0.6 patch).
- **Ordering = row-major:** `mask[ri*grid + ci]`, `ri`=row (y), `ci`=col (x). This is the
  standard ViT/DINOv2 token order and is **already relied on** by the validated multicolor mask.
  I add `analysis/verify_pusher_mask.py` to **confirm on a real frame** (overlays the dropped
  patches on `obs_g`; the zeroed patches must sit on the blue pusher).

## (d) Does pusher masking already exist? — YES, reuse it

- **Cost side is already built:** `objective_fn_last(..., vis_mask=None)` and
  `_masked_visual_mean` ([objectives.py:17-28](../planning/objectives.py#L17-L28)) already
  implement masked visual L2 (sum over KEPT patches / kept-count, so scale is comparable to
  unmasked). `objective_fn_all` too.
- **Plumbing is already built:** `CEMPlanner.patch_mask` (default `None`,
  [cem.py:45](../planning/cem.py#L45)) is sliced per-eval and passed as `vis_mask`
  ([cem.py:116-117](../planning/cem.py#L116-L117)).
- **Only the multicolor entry point sets it:** [plan_multicolor.py:50-55](../plan_multicolor.py#L50-L55)
  builds `pusher_patch_mask(state_g[:, :2])` and assigns `sub_planner.patch_mask`.
- **Gap:** the **stock** `plan.py` never sets `patch_mask`. STEP 2 = port that wiring to
  `PlanWorkspace`, as 2 Hydra flags, reusing `pusher_patch_mask` + the existing cost path.
  **No new cost math, no CEM-loop edit.**

## (e) Getting the PREDICTED pusher position (the both-sides subtlety) — and how I handle it

The pusher appears in **both** latents being differenced: the goal latent (goal pusher patches)
and the predicted last-frame latent (predicted pusher patches). Masking only the **goal** side
leaves the predicted-pusher patches compared against goal **background** → spurious error.

- The predicted final pusher xy is, in principle, recoverable, **but** (i) it is a property of
  each of the 300 CEM samples (and each opt step), and (ii) `cem.py` applies **one static
  `(n_evals, P)` mask** to all samples. Using a per-sample predicted-pusher mask would require
  editing the CEM loop — **forbidden**. Also `z_pred.proprio` is an *encoded* proprio embedding,
  not raw xy, so there is no cheap decode to patches.
- **Static proxy (the sanctioned fallback):** mask the **union of two known pusher poses** per
  eval, then optionally dilate:
  1. **goal-side** = the pusher actually rendered into `obs_g` (`state_g[:, :2]` for `real`;
     the fabricated `rsg[:, :2]` for `contact`).
  2. **rollout-side proxy** = the **real recorded pusher** `state_g[:, :2]`. For a *solvable*
     goal the predicted final pusher parks at the real contact configuration, so the real pusher
     is a good static stand-in for where the predicted pusher lands.
  - For `goal_pusher=real` the two coincide → the union is just the real pusher region (the
    both-sides issue is essentially **moot** for the linchpin N1 — clean measurement).
  - For `goal_pusher=contact` the union covers **both** the fabricated goal pusher **and** the
    real/predicted pusher → no spurious one-sided mismatch.
  - `mask_dilation` (0/1 rings) absorbs residual slack and tests contextual contamination of
    adjacent block patches.
- **Approximation logged** at runtime (the print states union + dilation + #patches dropped).

---

## Implementation plan (STEP 2), reusing everything above

Three Hydra knobs on the **stock** `plan.py` path (env=`pusht`):

| flag | type | meaning |
|---|---|---|
| `mask_pusher` | bool (false) | drop pusher patches from the visual L2 (sets `CEMPlanner.patch_mask`) |
| `mask_dilation` | int (0) | extra rings of neighbor patches to drop (0 = pusher patches only) |
| `objective.alpha` | float | **exists** — sweep {0,1}; `alpha=0` fully drops proprio |
| `goal_pusher_perturbation` | enum | **exists** — `real`\|`contact` IS flag #3 "goal_pusher_source" |

- New helper `manipulator_energy_mask(pusher_xys, dilation)` in `multicolor_common.py`
  (union of `pusher_patch_mask` over the xy list, then ring-dilation).
- `PlanWorkspace` stores `self.goal_pusher_xy` (rendered into `obs_g`) and `self.real_pusher_xy`
  (always real) in `prepare_targets`, and after building the planner sets
  `sub_planner.patch_mask` when `mask_pusher=true`. Guarded on env=`pusht` state layout
  `[ax, ay, bx, by, θ, vx, vy]` (pusher = `[0:2]`, block = `[2:5]`).
- **Untouched:** encoder, predictor/`rollout`, CEM search loop, success criterion
  (`pose_only_success` block-only, computed by the env independent of the energy).

## Matrix → flags (STEP 3)

All: `model_name=pusht goal_source=dset seed=99 n_evals=10 planner=cem opt_steps=30
num_samples=300 max_iter=10 pose_only_success=true`; same 10 goals.

| cond | alpha | mask_pusher | goal_pusher_perturbation | expect |
|---|---|---|---|---|
| R1 | 1 | false | real | ~1.0 ceiling |
| R2 | 1 | false | contact | ~0.6 |
| R3 | 0 | false | contact | ~0.3 |
| **N1** | **0** | **true** | **real** | **linchpin** |
| N2 | 0 | true | contact | g-deployment energy |
| N3 | 1 | true | contact | proprio-drag vs contamination |
| N4 | 0 | false | real | alpha=0 stall = shaping loss |
| N5(opt) | 0.1/0.3 | true | contact | weak shaping survives? |
| N6(opt) | 0 | true (dilation=1) | real | contamination sensitivity |
