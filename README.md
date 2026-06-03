# Cross-Modal Text-to-Goal Bridge for Latent World-Model Planning

Building **`g`**: a single-shot `(z_start, text) → z_goal` map that synthesizes a
per-patch goal latent in a **frozen** DINO-WM latent space, so the frozen DINO-WM
dynamics + CEM planner can reach a **language-specified** goal without ever being
shown a goal image. Testbed: a modified multi-color PushT.

> Status: Phase 0 (multi-color testbed + oracle ceiling). Phase 1 (`g`) is gated
> on the oracle reaching SR ≥ 0.80 on held-out color-location combos.

`dino_wm/` is a fork of [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)
(@ `0a9492f`), extended in place; the frozen encoder / dynamics / CEM are never modified.

## Phase 0 additions
```
dino_wm/
  env/pusht/
    multicolor_common.py        T geometry, palette, named-target coverage, pusher mask
    instructions.py             templated instructions (+ held-out template pool)
    multicolor_sampler.py       decorrelated layout sampler + color-location split
    pusht_multicolor_env.py     PushTEnv subclass: N colored T-outline decals (no physics)
    pusht_multicolor_wrapper.py gym entry point  (registered id: pusht_multicolor)
  conf/env/pusht_multicolor.yaml, conf/plan_pusht_multicolor.yaml
  datasets/pusht_multicolor_dset.py   trajectory loader + (z_start, instruction, z_goal) view
  scripts/                       gen_pusht_multicolor.py, cache_latents.py,
                                 verify_multicolor_env.py, setup_vastai.sh
  analysis/                      grounding_probe.py, pose_probe.py, dynamics_check.py
  plan_multicolor.py             oracle ceiling (reuses plan.py PlanWorkspace)
  planning/{objectives,cem}.py   optional manipulator-masked energy (backward compatible)
  tests/test_multicolor_env.py   unit tests (decorrelation, visual-only, named-target)
```

## Local sanity (CPU, Mac)
```bash
cd dino_wm
PYTHONPATH=. SDL_VIDEODRIVER=dummy python tests/test_multicolor_env.py        # 8/8
PYTHONPATH=. SDL_VIDEODRIVER=dummy python scripts/verify_multicolor_env.py    # render montage
```
The full pipeline (data gen → DINOv2 latent caching → probes → oracle planning)
runs on a single RTX 4090; `dino_wm/scripts/setup_vastai.sh` provisions the env.
