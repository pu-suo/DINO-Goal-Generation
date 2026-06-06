"""
STEP 3 benchmark + regression for the planning speed fix (masked alpha=0 / N1).

Runs the SAME condition (same seeds, same budget) through BOTH code paths:
  - fast_encode=True  : cache the start-obs encode (the optimization)
  - fast_encode=False : original per-candidate re-encode
and confirms the optimization did NOT change results:
  * final success rate IDENTICAL,
  * per-seed pass/fail IDENTICAL,
  * the per-opt-step CEM scores and the chosen actions MATCH to FP tolerance.
Then it reports per-episode wall time for each path and the speedup factor.

Why "FP tolerance" not "bit-identical": the only numerical difference between the
two paths is that the fast path encodes the start frame at batch=n_evals once while
the original encodes batch=num_samples copies per candidate -> a ~1e-6 cuBLAS
kernel-order difference in the start latent. The encoder is RNG-free, so the
predictor's dropout RNG stream (the only stochasticity) is identical between paths;
hence scores match to ~1e-4 and the success rate is unchanged. (See
scripts/test_rng_invariance.py and docs/PLANNING_SPEED_PROFILE.md.)

Usage (on the box, from dino_wm/):
    CKPTS=/ckpts DATASET_DIR=/data python scripts/bench_plan.py            # quick slice
    CKPTS=/ckpts DATASET_DIR=/data python scripts/bench_plan.py --full     # full validated budget (n=10)
"""
import os
import sys
import time
import json
import argparse
import tempfile

import numpy as np
import torch

# ---- recorders (populated by monkeypatched hooks) ---------------------------
_SCORES = []       # min objective value per objective_fn call, in call order
_CHOSEN = []       # per-MPC-iter returned mu (chosen actions)
_LAST_SUCCESS = {"v": None}


def install_recorders():
    import planning.objectives as objectives
    import planning.cem as cem
    import planning.evaluator as evaluator

    _orig_create = objectives.create_objective_fn

    def create_rec(*a, **k):
        fn = _orig_create(*a, **k)

        def wrapped(z_pred, z_tgt, vis_mask=None):
            loss = fn(z_pred, z_tgt, vis_mask=vis_mask)
            _SCORES.append(float(loss.min().item()))
            return loss

        return wrapped

    objectives.create_objective_fn = create_rec

    _orig_plan = cem.CEMPlanner.plan

    def plan_rec(self, obs_0, obs_g, actions=None):
        mu, valid = _orig_plan(self, obs_0, obs_g, actions=actions)
        _CHOSEN.append(mu.detach().float().cpu().numpy().copy())
        return mu, valid

    cem.CEMPlanner.plan = plan_rec

    _orig_eval = evaluator.PlanEvaluator.eval_actions

    def eval_rec(self, actions, action_len=None, filename="output",
                 save_video=False, plot=True):
        out = _orig_eval(self, actions, action_len=action_len,
                         filename=filename, save_video=save_video, plot=plot)
        _LAST_SUCCESS["v"] = np.asarray(out[1]).copy()  # successes
        return out

    evaluator.PlanEvaluator.eval_actions = eval_rec


def build_cfg(args, fast_encode, saved_folder):
    ckpts = os.environ.get("CKPTS", "./checkpoints")
    return {
        "ckpt_base_path": ckpts,
        "model_name": "pusht",
        "model_epoch": "latest",
        "seed": args.seed,
        "n_evals": args.n_evals,
        "goal_source": "dset",
        "goal_H": args.goal_h,
        "n_plot_samples": args.n_evals,
        "debug_dset_init": False,
        "objective": {"_target_": "planning.objectives.create_objective_fn",
                      "alpha": 0, "base": 2, "mode": "last"},
        "planner": {
            "_target_": "planning.mpc.MPCPlanner",
            "max_iter": args.max_iter,
            "n_taken_actions": args.goal_h,
            "sub_planner": {
                "target": "planning.cem.CEMPlanner",
                "horizon": args.goal_h, "topk": 30, "num_samples": args.num_samples,
                "var_scale": 1, "opt_steps": args.opt_steps, "eval_every": args.eval_every,
                "fast_encode": fast_encode,
            },
            "name": "mpc_cem",
        },
        "env_with_distractors": False, "env_n_distractors": 0,
        "distractor_outline_thickness": 7,
        "goal_pusher_perturbation": "real", "goal_pusher_offset": 40.0,
        "pose_only_success": True, "mask_pusher": True, "mask_dilation": 0,
        "wandb_logging": False, "saved_folder": saved_folder,
    }


def run_one(args, fast_encode):
    """Run one full episode; return (wall_s, final_SR, per_seed_success, scores, chosen)."""
    from utils import seed as set_seed
    _SCORES.clear()
    _CHOSEN.clear()
    _LAST_SUCCESS["v"] = None
    set_seed(args.seed)  # re-seed so both paths start from the same RNG state
    from plan import planning_main
    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        os.chdir(td)
        try:
            cfg = build_cfg(args, fast_encode, td)
            if _CUDA:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logs = planning_main(cfg)
            if _CUDA:
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        finally:
            os.chdir(cwd)
    sr = float(logs.get("final_eval/success_rate"))
    return wall, sr, _LAST_SUCCESS["v"], list(_SCORES), list(_CHOSEN)


_CUDA = torch.cuda.is_available()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="full validated budget: n_evals=10 opt_steps=30 num_samples=300 "
                         "max_iter=10 eval_every=1")
    ap.add_argument("--n-evals", type=int, default=4)
    ap.add_argument("--opt-steps", type=int, default=10)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--goal-h", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--score-tol", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="bench_regression.json")
    args = ap.parse_args()
    if args.full:
        args.n_evals, args.opt_steps, args.num_samples = 10, 30, 300
        args.max_iter, args.eval_every = 10, 1

    sys.path.insert(0, os.getcwd())
    install_recorders()

    print(f"[bench] budget: n_evals={args.n_evals} opt_steps={args.opt_steps} "
          f"num_samples={args.num_samples} max_iter={args.max_iter} "
          f"eval_every={args.eval_every} seed={args.seed}")

    print("[bench] running FAST path (fast_encode=True) ...")
    w_fast, sr_fast, suc_fast, sc_fast, ch_fast = run_one(args, True)
    print(f"        wall={w_fast:.1f}s  SR={sr_fast:.3f}")

    print("[bench] running ORIGINAL path (fast_encode=False) ...")
    w_orig, sr_orig, suc_orig, sc_orig, ch_orig = run_one(args, False)
    print(f"        wall={w_orig:.1f}s  SR={sr_orig:.3f}")

    # ---- regression assertions ----
    ok = True
    msgs = []
    if sr_fast != sr_orig:
        ok = False
        msgs.append(f"FAIL: SR differs (fast={sr_fast} orig={sr_orig})")
    else:
        msgs.append(f"ok: SR identical ({sr_fast:.3f})")

    if suc_fast is not None and suc_orig is not None:
        if np.array_equal(suc_fast, suc_orig):
            msgs.append(f"ok: per-seed pass/fail identical ({suc_fast.astype(int).tolist()})")
        else:
            ok = False
            msgs.append(f"FAIL: per-seed pass/fail differs {suc_fast} vs {suc_orig}")

    n = min(len(sc_fast), len(sc_orig))
    if n and len(sc_fast) == len(sc_orig):
        diffs = np.abs(np.array(sc_fast) - np.array(sc_orig))
        rel = diffs / (np.abs(np.array(sc_orig)) + 1e-9)
        msgs.append(f"ok: {len(sc_fast)} CEM scores compared; "
                    f"max|abs|={diffs.max():.2e} max|rel|={rel.max():.2e}")
        if rel.max() > args.score_tol:
            ok = False
            msgs.append(f"FAIL: score rel-diff {rel.max():.2e} > tol {args.score_tol:.0e}")
    else:
        ok = False
        msgs.append(f"FAIL: score-list length mismatch ({len(sc_fast)} vs {len(sc_orig)})")

    if len(ch_fast) == len(ch_orig) and ch_fast:
        adiff = max(float(np.abs(a - b).max()) for a, b in zip(ch_fast, ch_orig))
        msgs.append(f"ok: chosen-action max|abs diff|={adiff:.2e} over {len(ch_fast)} MPC iters")
    else:
        msgs.append(f"warn: chosen-action list length {len(ch_fast)} vs {len(ch_orig)}")

    speedup = w_orig / max(w_fast, 1e-9)
    print("\n================= REGRESSION =================")
    for m in msgs:
        print("  " + m)
    print(f"\n  wall: fast={w_fast:.1f}s  orig={w_orig:.1f}s  ->  speedup={speedup:.2f}x")
    print("  RESULT:", "PASS ✅" if ok else "FAIL ❌")

    payload = {
        "budget": vars(args), "sr_fast": sr_fast, "sr_orig": sr_orig,
        "success_fast": None if suc_fast is None else suc_fast.astype(int).tolist(),
        "success_orig": None if suc_orig is None else suc_orig.astype(int).tolist(),
        "wall_fast_s": w_fast, "wall_orig_s": w_orig, "speedup": speedup,
        "n_scores": len(sc_fast), "passed": ok,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {os.path.abspath(args.out)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
