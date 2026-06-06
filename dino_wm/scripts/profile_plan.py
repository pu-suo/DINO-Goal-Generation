"""
STEP 1 profiler: wall-time breakdown of ONE planning episode (masked alpha=0 / N1).

Runs on the GPU box (needs $CKPTS with outputs/pusht and $DATASET_DIR/pusht_noise).
It instruments the masked alpha=0 condition and reports where time goes across:
  (a) dynamics-model forward (the latent rollout)  -> VWorldModel.predict
  (a')start/goal encode (DINOv2)                    -> VWorldModel.encode_obs
  (b) environment stepping (pymunk + render)        -> SubprocVectorEnv.rollout
  (c) decoder (VQ-VAE) for the debug plots          -> VWorldModel.decode_obs
  (d) CEM scoring (objective_fn)                    -> objective closure
  (e/f) everything else (CEM bookkeeping: sampling, argsort, refit; CPU<->GPU
        transfers; python/loop overhead)            -> wall - sum(above)

Defaults to a SHORT slice (max_iter=1, opt_steps=6) so it finishes fast; the
per-category PROPORTIONS are representative of the full budget. Bump with flags
for a full-length profile.

Usage (on the box, from dino_wm/):
    CKPTS=/ckpts DATASET_DIR=/data python scripts/profile_plan.py
    CKPTS=/ckpts DATASET_DIR=/data python scripts/profile_plan.py --opt-steps 30 --max-iter 1 --n-evals 4
    # profile the ORIGINAL (per-candidate re-encode) path instead of the fast one:
    CKPTS=/ckpts DATASET_DIR=/data python scripts/profile_plan.py --no-fast-encode
"""
import os
import sys
import time
import argparse
import tempfile
from collections import OrderedDict

import torch

# ---- timing registry --------------------------------------------------------
_T = OrderedDict()  # name -> [total_seconds, n_calls]
_CUDA = torch.cuda.is_available()


def _sync():
    if _CUDA:
        torch.cuda.synchronize()


def timed(name):
    """Wrap a callable so its (GPU-synced) wall time accumulates under `name`."""
    def deco(fn):
        def wrapper(*a, **k):
            _sync()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            _sync()
            _T.setdefault(name, [0.0, 0])
            _T[name][0] += time.perf_counter() - t0
            _T[name][1] += 1
            return out
        return wrapper
    return deco


def install_instrumentation():
    """Monkeypatch the hot methods at class level (planning_main builds the
    instances internally, so class-level patching catches them)."""
    from models.visual_world_model import VWorldModel
    from env.venv import SubprocVectorEnv
    import planning.objectives as objectives

    VWorldModel.encode_obs = timed("a' encode (DINOv2)")(VWorldModel.encode_obs)
    VWorldModel.predict = timed("a  dynamics predict")(VWorldModel.predict)
    if VWorldModel.decode_obs is not None:
        VWorldModel.decode_obs = timed("c  decoder (VQ-VAE)")(VWorldModel.decode_obs)
    SubprocVectorEnv.rollout = timed("b  env rollout (sim+render)")(SubprocVectorEnv.rollout)

    _orig_create = objectives.create_objective_fn

    def _wrapped_create(*a, **k):
        fn = _orig_create(*a, **k)
        return timed("d  CEM scoring (objective)")(fn)

    objectives.create_objective_fn = _wrapped_create


def build_cfg(args):
    """Build the plan_pusht cfg_dict for the masked alpha=0 (N1) condition, mirroring
    analysis/run_masked_energy_matrix.sh N1, with a (default short) profiling budget."""
    ckpts = os.environ.get("CKPTS", "./checkpoints")
    sub_planner = {
        "target": "planning.cem.CEMPlanner",
        "horizon": args.goal_h,
        "topk": 30,
        "num_samples": args.num_samples,
        "var_scale": 1,
        "opt_steps": args.opt_steps,
        "eval_every": args.eval_every,
        "fast_encode": args.fast_encode,
    }
    cfg = {
        "ckpt_base_path": ckpts,
        "model_name": "pusht",
        "model_epoch": "latest",
        "seed": args.seed,
        "n_evals": args.n_evals,
        "goal_source": "dset",
        "goal_H": args.goal_h,
        "n_plot_samples": args.n_evals,
        "debug_dset_init": False,
        "objective": {
            "_target_": "planning.objectives.create_objective_fn",
            "alpha": 0,
            "base": 2,
            "mode": "last",
        },
        "planner": {
            "_target_": "planning.mpc.MPCPlanner",
            "max_iter": args.max_iter,
            "n_taken_actions": args.goal_h,
            "sub_planner": sub_planner,
            "name": "mpc_cem",
        },
        # isolation / masked-energy knobs (N1: alpha=0, mask_pusher=true, real pusher)
        "env_with_distractors": False,
        "env_n_distractors": 0,
        "distractor_outline_thickness": 7,
        "goal_pusher_perturbation": args.goal_pusher,
        "goal_pusher_offset": 40.0,
        "pose_only_success": True,
        "mask_pusher": True,
        "mask_dilation": 0,
        "wandb_logging": False,
    }
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-evals", type=int, default=4)
    ap.add_argument("--opt-steps", type=int, default=6)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=1)
    ap.add_argument("--goal-h", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="match the validated default (1). Set high to skip inner evals.")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--goal-pusher", type=str, default="real")
    ap.add_argument("--no-fast-encode", dest="fast_encode", action="store_false")
    ap.set_defaults(fast_encode=True)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    install_instrumentation()
    from plan import planning_main

    cfg = build_cfg(args)
    with tempfile.TemporaryDirectory() as td:
        cfg["saved_folder"] = td
        os.chdir(td)  # plan dumps logs/pngs into cwd
        print(f"[profile] fast_encode={args.fast_encode} n_evals={args.n_evals} "
              f"opt_steps={args.opt_steps} num_samples={args.num_samples} "
              f"max_iter={args.max_iter} eval_every={args.eval_every}")
        _sync()
        t0 = time.perf_counter()
        planning_main(cfg)
        _sync()
        wall = time.perf_counter() - t0

    print("\n================= PROFILE (one episode) =================")
    accounted = sum(v[0] for v in _T.values())
    rows = sorted(_T.items(), key=lambda kv: -kv[1][0])
    print(f"{'category':<32}{'sec':>9}{'%wall':>8}{'calls':>9}{'ms/call':>10}")
    print("-" * 68)
    for name, (sec, n) in rows:
        print(f"{name:<32}{sec:>9.2f}{100*sec/wall:>7.1f}%{n:>9}{1000*sec/max(n,1):>9.2f}")
    other = wall - accounted
    print(f"{'e/f bookkeeping+transfer+python':<32}{other:>9.2f}{100*other/wall:>7.1f}%{'':>9}{'':>10}")
    print("-" * 68)
    print(f"{'TOTAL WALL':<32}{wall:>9.2f}{100.0:>7.1f}%")
    print("\nNote: short-slice proportions are representative; scale by "
          "(max_iter*opt_steps) for the full-budget wall.")


if __name__ == "__main__":
    main()
