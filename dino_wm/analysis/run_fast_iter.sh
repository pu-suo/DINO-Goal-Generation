#!/usr/bin/env bash
# STEP 4 -- fast-iteration runner for the masked alpha=0 (N1) condition.
#
# This is the cheap-iteration config: a SINGLE condition with the search budget and
# n_evals trivially overridable, so you can iterate quickly at a reduced budget while
# keeping the FULL budget for final numbers. It deliberately uses:
#   * fast_encode=true        (the result-preserving speed fix; default on)
#   * eval_every=999          (one SR/iter; inner-eval cadence -- plan-quality-neutral
#                              per Phase-0, see docs/PHASE0_ISOLATION_HANDOFF.md)
#
# IMPORTANT: reducing OPT_STEPS / NUM_SAMPLES is an accuracy<->speed trade for
# ITERATION ONLY. It is NOT part of the regression-tested speed fix (which runs the
# full budget). For final/headline numbers use the full budget (OPT_STEPS=30
# NUM_SAMPLES=300 N_EVALS>=10) and eval_every from the validated config.
#
# Usage (from dino_wm/):
#   CKPTS=/ckpts DATASET_DIR=/data bash analysis/run_fast_iter.sh
#   OPT_STEPS=15 NUM_SAMPLES=200 N_EVALS=10 CKPTS=/ckpts DATASET_DIR=/data \
#       bash analysis/run_fast_iter.sh
set -u
cd "$(dirname "$0")/.." || exit 1

CKPTS="${CKPTS:-./checkpoints}"
PY="${PY:-python}"
N_EVALS="${N_EVALS:-10}"
OPT_STEPS="${OPT_STEPS:-15}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
MAX_ITER="${MAX_ITER:-10}"
SEED="${SEED:-99}"
EVAL_EVERY="${EVAL_EVERY:-999}"
FAST_ENCODE="${FAST_ENCODE:-true}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DIR="plan_outputs/fast_iter/$STAMP"
mkdir -p "$DIR"

echo "=== fast-iter N1 (masked alpha=0): opt_steps=$OPT_STEPS num_samples=$NUM_SAMPLES "\
"n_evals=$N_EVALS eval_every=$EVAL_EVERY fast_encode=$FAST_ENCODE -> $DIR ==="

$PY plan.py --config-name plan_pusht \
    model_name=pusht ckpt_base_path="$CKPTS" \
    goal_source=dset seed="$SEED" n_evals="$N_EVALS" goal_H=5 \
    pose_only_success=true \
    objective.alpha=0 mask_pusher=true mask_dilation=0 \
    goal_pusher_perturbation=real \
    planner.max_iter="$MAX_ITER" \
    planner.sub_planner.opt_steps="$OPT_STEPS" \
    planner.sub_planner.num_samples="$NUM_SAMPLES" \
    planner.sub_planner.eval_every="$EVAL_EVERY" \
    +planner.sub_planner.fast_encode="$FAST_ENCODE" \
    hydra.run.dir="$DIR" 2>&1 | tee "$DIR/run.log"

echo "SR: $(grep -h 'Success rate:' "$DIR/run.log" | tail -1 | sed 's/.*Success rate: *//')"
