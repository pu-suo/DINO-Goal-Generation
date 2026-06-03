# Cross-Modal Text-to-Goal Bridge for Latent World-Model Planning

Building **`g` (the "bridge")**: a single-shot `(z_start, text) → z_goal` map that
synthesizes a per-patch goal latent in a **frozen** DINO-WM latent space, so the
frozen DINO-WM dynamics + CEM planner can reach a **language-specified** goal
**without ever being shown a goal image**. Testbed: a modified multi-color PushT.

- **Architecture, mental model, hard rules:** [`CLAUDE.md`](CLAUDE.md)
- **Phase 0 plan / gates:** [`specs/PHASE_0_PLAN.md`](specs/PHASE_0_PLAN.md)
- **Phase 0 run order (vast.ai):** [`specs/PHASE_0_RUNBOOK.md`](specs/PHASE_0_RUNBOOK.md)
- **Mac↔box sync:** [`SYNC.md`](SYNC.md)
- **Upstream provenance:** [`dino_wm/UPSTREAM.md`](dino_wm/UPSTREAM.md)

> Status: **Phase 0 in progress.** `g` (Phase 1) is gated on Phase 0, especially
> oracle SR ≥ 0.80 on held-out color-location combos.

## Layout
`dino_wm/` is our fork of [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)
(@ `0a9492f`), extended in place; the frozen encoder/dynamics/CEM are never modified.
Phase 0 additions (all new files unless noted):

```
dino_wm/
  env/pusht/
    multicolor_common.py        T geometry, palette, named-target coverage, pusher mask
    instructions.py             templated instructions (+ held-out template pool)
    multicolor_sampler.py       decorrelated layout sampler + color-location split
    pusht_multicolor_env.py     PushTEnv subclass: N colored T-outline decals (no physics)
    pusht_multicolor_wrapper.py gym entry point
    __init__.py (edit)          registers `pusht_multicolor`; makes pointmaze import optional
  conf/env/pusht_multicolor.yaml, conf/plan_pusht_multicolor.yaml
  datasets/pusht_multicolor_dset.py   trajectory loader + (z_start, instruction, z_goal) view
  scripts/gen_pusht_multicolor.py     dataset generation (CPU-parallel)
  scripts/cache_latents.py            frozen DINOv2 latent caching
  scripts/verify_multicolor_env.py    local env preview
  scripts/setup_vastai.sh             box environment setup
  analysis/grounding_probe.py, pose_probe.py, dynamics_check.py, probe_common.py
  plan_multicolor.py                  oracle ceiling (reuses plan.py PlanWorkspace)
  planning/objectives.py, cem.py (edits)  optional manipulator-masked energy (backward compatible)
  tests/test_multicolor_env.py        8 unit tests (decorrelation, visual-only, named-target)
```

## Quickstart (local dev, Mac)
Local env `dino_wm_dev` (py3.10) — invoke via its absolute interpreter
`/Users/Tom/miniforge3/envs/dino_wm_dev/bin/python` (NOT `conda run`).
```bash
cd dino_wm
PYTHONPATH=. SDL_VIDEODRIVER=dummy <dino_wm_dev python> tests/test_multicolor_env.py     # 8/8
PYTHONPATH=. SDL_VIDEODRIVER=dummy <dino_wm_dev python> scripts/verify_multicolor_env.py # montage
```
For the full pipeline (data gen → caching → probes → oracle) on the 4090, follow
[`specs/PHASE_0_RUNBOOK.md`](specs/PHASE_0_RUNBOOK.md).
