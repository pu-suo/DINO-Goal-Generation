#!/usr/bin/env bash
# STEP 4 -- budget knee sweep for the masked alpha=0 (N1) condition.
#
# Sweeps opt_steps in {10,15,20,30} x num_samples in {100,200,300} at n_evals=10 and
# tabulates final-eval SR, so you can pick the smallest budget where SR stops moving
# (the "knee") for cheap iteration -- while keeping the full budget (30 x 300) for
# final numbers. This is an iteration speed<->accuracy study, SEPARATE from the
# result-preserving speed fix.
#
# Uses the fast (result-preserving) encode path and eval_every=999 (plan-quality-
# neutral inner-eval cadence per Phase-0) so the whole grid runs as fast as possible.
#
# Usage (from dino_wm/):
#   CKPTS=/ckpts DATASET_DIR=/data bash analysis/sweep_budget.sh
#   N_EVALS=10 OPT_STEPS_LIST="10 20 30" NUM_SAMPLES_LIST="100 300" \
#       CKPTS=/ckpts DATASET_DIR=/data bash analysis/sweep_budget.sh
set -u
cd "$(dirname "$0")/.." || exit 1

CKPTS="${CKPTS:-./checkpoints}"
PY="${PY:-python}"
N_EVALS="${N_EVALS:-10}"
MAX_ITER="${MAX_ITER:-10}"
SEED="${SEED:-99}"
OPT_STEPS_LIST="${OPT_STEPS_LIST:-10 15 20 30}"
NUM_SAMPLES_LIST="${NUM_SAMPLES_LIST:-100 200 300}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTROOT="plan_outputs/budget_sweep/$STAMP"
mkdir -p "$OUTROOT"

echo "=== budget knee sweep (N1 masked alpha=0), n_evals=$N_EVALS, seed=$SEED ==="
echo "opt_steps in {$OPT_STEPS_LIST} x num_samples in {$NUM_SAMPLES_LIST} -> $OUTROOT"

for OS in $OPT_STEPS_LIST; do
  for NS in $NUM_SAMPLES_LIST; do
    DIR="$OUTROOT/os${OS}_ns${NS}"
    mkdir -p "$DIR"
    echo "--- opt_steps=$OS num_samples=$NS ---"
    $PY plan.py --config-name plan_pusht \
        model_name=pusht ckpt_base_path="$CKPTS" \
        goal_source=dset seed="$SEED" n_evals="$N_EVALS" goal_H=5 \
        pose_only_success=true \
        objective.alpha=0 mask_pusher=true mask_dilation=0 \
        goal_pusher_perturbation=real \
        planner.max_iter="$MAX_ITER" \
        planner.sub_planner.opt_steps="$OS" \
        planner.sub_planner.num_samples="$NS" \
        planner.sub_planner.eval_every=999 \
        +planner.sub_planner.fast_encode=true \
        hydra.run.dir="$DIR" 2>&1 | tee "$DIR/run.log" >/dev/null
    sr=$(grep -h 'Success rate:' "$DIR/run.log" | tail -1 | sed 's/.*Success rate: *//')
    echo "    SR=${sr:-NA}"
  done
done

echo ""
echo "===================== SR KNEE TABLE (n=$N_EVALS) ====================="
printf "%-12s" "opt\\ns"
for NS in $NUM_SAMPLES_LIST; do printf "%-8s" "$NS"; done
echo
for OS in $OPT_STEPS_LIST; do
  printf "%-12s" "$OS"
  for NS in $NUM_SAMPLES_LIST; do
    sr=$(grep -h 'Success rate:' "$OUTROOT/os${OS}_ns${NS}/run.log" 2>/dev/null | tail -1 | sed 's/.*Success rate: *//')
    printf "%-8s" "${sr:-NA}"
  done
  echo
done
echo "Pick the smallest (opt_steps, num_samples) where SR plateaus = the knee."
