#!/usr/bin/env bash
# Three-way quasimetric eval on the STOCK single-T pusht model (the decision run).
# Question: does the learned asymmetric cost-to-go (NEW) close the FLOOR->CEILING gap?
#
#   FLOOR   : masked alpha=0, masked-L2 only            (the existing N1/N2 baseline)
#   NEW     : masked alpha=0, quasimetric + terminal-L2 (this work)
#   CEILING : alpha=1, real pusher, unmasked            (privileged proprio upper bound)
#
# All three run on the IDENTICAL genuine held-out goal set: goal_source=dset (real val
# trajectory segments -- the standard DINO-WM PushT eval, == HWM d=25), seed fixed so the
# sampled goals match across conditions, pose-only success (block pos<20px AND ang<pi/9),
# REGULAR planning budget (opt_steps=30 num_samples=300 max_iter=10 goal_H=5). We do NOT
# use any easy subset: the multicolor max_goal_dist/max_goal_angle crutches are
# multicolor-only and are not on this stock single-T path. Re-run all three together so
# they are directly comparable (do not reuse old numbers).
#
# Usage:
#   QM_CKPT=$CKPTS/qm/iqe_d0/qm_head.pth bash analysis/run_qm_eval.sh all
#   QM_CKPT=... N_EVALS=50 W_QM=1.0 W_L2=10.0 bash analysis/run_qm_eval.sh floor new ceiling
#   QM_CKPT=... bash analysis/run_qm_eval.sh sweep          # w_qm:w_l2 knee sweep (NEW only)
set -u
cd "$(dirname "$0")/.." || exit 1

CKPTS="${CKPTS:-./checkpoints}"
QM_CKPT="${QM_CKPT:-}"
N_EVALS="${N_EVALS:-50}"
SEED="${SEED:-99}"
W_QM="${W_QM:-1.0}"
W_L2="${W_L2:-10.0}"
PER_STEP="${PER_STEP:-false}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTROOT="plan_outputs/qm_eval/$STAMP"
PY="${PY:-python}"

# Regular budget, matched to the masked-energy matrix that produced the FLOOR/CEILING
# anchors (so the comparison is apples-to-apples).
COMMON=(--config-name plan_pusht
        model_name=pusht ckpt_base_path="$CKPTS"
        goal_source=dset seed="$SEED" n_evals="$N_EVALS" goal_H=5
        pose_only_success=true
        planner.max_iter=10
        planner.sub_planner.opt_steps=30
        planner.sub_planner.num_samples=300)

run_floor () {
  local dir="$OUTROOT/floor"; mkdir -p "$dir"
  echo "=== FLOOR: masked alpha=0, L2-only -> $dir ==="
  $PY plan.py "${COMMON[@]}" objective.alpha=0 mask_pusher=true \
      goal_pusher_perturbation=real hydra.run.dir="$dir" 2>&1 | tee "$dir/floor.log"
}
run_ceiling () {
  local dir="$OUTROOT/ceiling"; mkdir -p "$dir"
  echo "=== CEILING: alpha=1, real pusher, unmasked -> $dir ==="
  $PY plan.py "${COMMON[@]}" objective.alpha=1 mask_pusher=false \
      goal_pusher_perturbation=real hydra.run.dir="$dir" 2>&1 | tee "$dir/ceiling.log"
}
run_new () {
  [[ -z "$QM_CKPT" ]] && { echo "set QM_CKPT=<path to qm_head.pth>"; return 1; }
  local dir="$OUTROOT/new"; mkdir -p "$dir"
  echo "=== NEW: masked alpha=0, qm($W_QM)+L2($W_L2) per_step=$PER_STEP -> $dir ==="
  $PY plan.py "${COMMON[@]}" objective.alpha=0 mask_pusher=true \
      goal_pusher_perturbation=real \
      qm.enabled=true qm.ckpt="$QM_CKPT" qm.w_qm="$W_QM" qm.w_l2="$W_L2" qm.per_step="$PER_STEP" \
      hydra.run.dir="$dir" 2>&1 | tee "$dir/new.log"
}
run_sweep () {
  [[ -z "$QM_CKPT" ]] && { echo "set QM_CKPT=<path to qm_head.pth>"; return 1; }
  for wq in 0.3 1.0 3.0; do for wl in 1.0 10.0; do
    local dir="$OUTROOT/sweep_wq${wq}_wl${wl}"; mkdir -p "$dir"
    echo "=== SWEEP w_qm=$wq w_l2=$wl -> $dir ==="
    $PY plan.py "${COMMON[@]}" objective.alpha=0 mask_pusher=true goal_pusher_perturbation=real \
        qm.enabled=true qm.ckpt="$QM_CKPT" qm.w_qm="$wq" qm.w_l2="$wl" \
        hydra.run.dir="$dir" 2>&1 | tee "$dir/sweep.log"
  done; done
}

sel=("$@")
[[ ${#sel[@]} -eq 0 ]] && { echo "give conditions: floor new ceiling | all | sweep"; exit 1; }
[[ "${sel[0]}" == "all" ]] && sel=(floor new ceiling)
for c in "${sel[@]}"; do
  case "$c" in
    floor) run_floor;; new) run_new;; ceiling) run_ceiling;; sweep) run_sweep;;
    *) echo "unknown: $c";;
  esac
done

echo ""
echo "===================== SUMMARY (final-eval SR, n=$N_EVALS, seed=$SEED) ====================="
for log in "$OUTROOT"/*/*.log; do
  [[ -e "$log" ]] || continue
  cond=$(basename "$(dirname "$log")")
  sr=$(grep -h "Success rate:" "$log" 2>/dev/null | tail -1 | sed 's/.*Success rate: *//')
  printf "%-28s SR=%s\n" "$cond" "${sr:-NA}"
done
echo "Decision: does NEW close most of the FLOOR->CEILING gap on these genuine goals?"
