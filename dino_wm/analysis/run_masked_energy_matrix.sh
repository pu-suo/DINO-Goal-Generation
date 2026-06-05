#!/usr/bin/env bash
# Masked-energy experiment matrix on the STOCK pusht model (docs/MASKED_ENERGY_NOTE.md).
# Runs SEQUENTIALLY (4 concurrent CEM runs OOM a 24GB 4090). Same 10 goals everywhere.
#
# Usage:
#   bash analysis/run_masked_energy_matrix.sh N1                 # validate the linchpin first
#   bash analysis/run_masked_energy_matrix.sh R1 R2 R3 N1 N2 N3 N4   # the full matrix
#   bash analysis/run_masked_energy_matrix.sh all                # R1..N4 in order
#   STAMP=myrun bash analysis/run_masked_energy_matrix.sh N1     # custom output subdir
#
# Each condition -> plan_outputs/masked_matrix/$STAMP/<COND>/ (+ <COND>.log). The final
# "Success rate:" line of each log is the headline SR over n=10.
set -u
cd "$(dirname "$0")/.." || exit 1

CKPTS="${CKPTS:-./checkpoints}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTROOT="plan_outputs/masked_matrix/$STAMP"
PY="${PY:-python}"

# matched settings from the prior Phase-0 finding (stock pusht real-goal == SR 1.0)
COMMON=(--config-name plan_pusht
        model_name=pusht ckpt_base_path="$CKPTS"
        goal_source=dset seed=99 n_evals=10 goal_H=5
        pose_only_success=true
        planner.max_iter=10
        planner.sub_planner.opt_steps=30
        planner.sub_planner.num_samples=300)

# cond -> "alpha mask_pusher goal_pusher_perturbation mask_dilation"
declare -A COND=(
  [R1]="1 false real 0"      # ceiling           expect ~1.0
  [R2]="1 false contact 0"   # fabricated pusher  expect ~0.6
  [R3]="0 false contact 0"   # alpha=0 unmasked   expect ~0.3
  [N1]="0 true  real 0"      # LINCHPIN: masked object-only L2, perfect goal
  [N2]="0 true  contact 0"   # real g-deployment energy
  [N3]="1 true  contact 0"   # proprio-drag vs visual contamination
  [N4]="0 false real 0"      # alpha=0 unmasked, real goal (shaping-loss control)
  [N5a]="0.1 true contact 0" # weak shaping (optional)
  [N5b]="0.3 true contact 0" # weak shaping (optional)
  [N6]="0 true  real 1"      # N1 with dilation=1 (contamination sensitivity, optional)
)
ORDER=(R1 R2 R3 N1 N2 N3 N4)

run_cond () {
  local c="$1"
  local spec="${COND[$c]:-}"
  if [[ -z "$spec" ]]; then echo "unknown condition: $c"; return 1; fi
  read -r alpha mask gpp dil <<< "$spec"
  local dir="$OUTROOT/$c"
  mkdir -p "$dir"
  echo "=== $c : alpha=$alpha mask_pusher=$mask goal_pusher=$gpp dilation=$dil -> $dir ==="
  $PY plan.py "${COMMON[@]}" \
      objective.alpha="$alpha" \
      mask_pusher="$mask" mask_dilation="$dil" \
      goal_pusher_perturbation="$gpp" \
      hydra.run.dir="$dir" 2>&1 | tee "$dir/$c.log"
}

sel=("$@")
[[ ${#sel[@]} -eq 0 ]] && { echo "give conditions, e.g. N1  (or 'all')"; exit 1; }
[[ "${sel[0]}" == "all" ]] && sel=("${ORDER[@]}")

for c in "${sel[@]}"; do run_cond "$c"; done

echo ""
echo "===================== SUMMARY (final-eval SR over n=10) ====================="
printf "%-5s %-7s %-6s %-9s %-4s  %s\n" COND alpha mask goalpush dil SR
for c in "${sel[@]}"; do
  read -r alpha mask gpp dil <<< "${COND[$c]}"
  sr=$(grep -h "Success rate:" "$OUTROOT/$c/$c.log" 2>/dev/null | tail -1 | sed 's/.*Success rate: *//')
  printf "%-5s %-7s %-6s %-9s %-4s  %s\n" "$c" "$alpha" "$mask" "$gpp" "$dil" "${sr:-NA}"
done
