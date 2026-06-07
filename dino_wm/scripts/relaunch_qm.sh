#!/usr/bin/env bash
# ============================================================================
# ONE-SHOT quasimetric relaunch -- fresh vast.ai instance -> decision numbers.
# ============================================================================
# Runs the WHOLE pipeline hands-off and saves EVERY log under one dir (the box
# can't scroll up). Idempotent: each stage is skipped if its output already
# exists, so a re-run after a crash/Stop-Start resumes where it left off.
#
#   0. env      setup_vastai.sh   (only if $WS/activate.sh is missing)
#   1. data     download_data.sh  (only if pusht_noise is missing)
#   2. cache    model-step latents (train + val)              [skip if cached]
#   3. train    calibrated iqe_d1 head + LIVE val monitor     [skip if trained]
#   4. validate gates (a)(b)(c) on held-out val, on the BEST-VAL head
#   5. eval     3-way FLOOR / NEW / CEILING on genuine held-out single-T goals
#
# This is the recalibrated retrain after iqe_d0 FAILED its gates by SILENT
# OVERFITTING (held-out adjacent-d ~9x train; d_sg inflated to ~108). The fixes,
# all baked in as defaults below: sharper phi (beta 0.1->0.5) + bounded goals
# (max_goal_offset 32) to stop d_sg inflation; more data + fewer steps +
# cross-traj goals + weight decay to curb overfitting; and the train script now
# prints the train/val gap LIVE and keeps qm_head_bestval.pth.
#
# BOOTSTRAP on a brand-new instance (do this BEFORE running the script so the
# script + code are the current version; do NOT git pull from inside the script):
#   cd /workspace
#   git clone https://github.com/pu-suo/DINO-Goal-Generation.git dino_goal \
#       || (cd dino_goal && git pull)
#   cd dino_goal/dino_wm
#   tmux new-session -d -s qm && tmux switch-client -t qm   # (if already in tmux)
#   bash scripts/relaunch_qm.sh
#
# Tunables (env overrides, e.g. `N_TRAIN=8000 bash scripts/relaunch_qm.sh`):
#   TAG, N_TRAIN, N_VAL, MAX_GB, STEPS, PHI_OFFSET, PHI_BETA, MAX_GOAL_OFFSET,
#   P_RANDOM_GOAL, WEIGHT_DECAY, NUM_WORKERS, N_EVALS, W_QM, W_L2, DO_SWEEP,
#   FORCE_CACHE=1, FORCE_TRAIN=1   (bust the skip-if-exists guards)
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1          # -> dino_wm

WS=${WS:-/workspace}

# ---- 0. env -----------------------------------------------------------------
if [[ ! -f "$WS/activate.sh" ]]; then
  echo "==> [0] no $WS/activate.sh -- running setup_vastai.sh (one-time)"
  bash scripts/setup_vastai.sh
fi
# shellcheck disable=SC1090
source "$WS/activate.sh"
: "${DATASET_DIR:?activate.sh did not set DATASET_DIR -- setup failed}"
: "${CKPTS:?activate.sh did not set CKPTS -- setup failed}"

# ---- params (calibrated iqe_d1 defaults) ------------------------------------
TAG=${TAG:-iqe_d1}
N_TRAIN=${N_TRAIN:-6000}        # ~22 GB f16 cache (& ~that much RAM). 8000+ on a big-RAM box.
N_VAL=${N_VAL:-400}
MAX_GB=${MAX_GB:-35}            # cache disk/RAM guard (raise with N_TRAIN)
STEPS=${STEPS:-20000}          # ~26 epochs over 6k trajs (iqe_d0 ran ~210 -> overfit)
PHI_OFFSET=${PHI_OFFSET:-25}   # target cost-to-go scale; just under MAX_GOAL_OFFSET
PHI_BETA=${PHI_BETA:-0.5}      # sharper than iqe_d0's 0.1 (which let d_sg inflate to 108)
MAX_GOAL_OFFSET=${MAX_GOAL_OFFSET:-32}   # bound value-pair goal distance (model-steps)
P_RANDOM_GOAL=${P_RANDOM_GOAL:-0.3}      # cross-traj goals (QRL spreading / generalization)
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}       # small AdamW WD -- anti-overfit belt
NUM_WORKERS=${NUM_WORKERS:-4}            # mask build was the iqe_d0 CPU bottleneck
N_EVALS=${N_EVALS:-30}                   # paired across FLOOR/NEW/CEILING (project's ">=30")
W_QM=${W_QM:-1.0}
W_L2=${W_L2:-10.0}
DO_SWEEP=${DO_SWEEP:-1}                   # cheap w_qm:w_l2 knee sweep (n=10) before the headline
FORCE_CACHE=${FORCE_CACHE:-0}
FORCE_TRAIN=${FORCE_TRAIN:-0}

STAMP=$(date +%Y%m%d_%H%M%S)
RUN="$WS/qm_runs/${TAG}_${STAMP}"
mkdir -p "$RUN"
CACHE="$DATASET_DIR/pusht_noise/qm_latents"
OUT="$CKPTS/qm/$TAG"

# Mirror ALL stdout/stderr to one master log (the box can't scroll up).
exec > >(tee -a "$RUN/relaunch.log") 2>&1
echo "================================================================"
echo " QM RELAUNCH  tag=$TAG  $(date)"
echo " run dir : $RUN     (everything is logged here)"
echo " cache   : $CACHE   head out: $OUT"
echo " train   : N_TRAIN=$N_TRAIN steps=$STEPS phi_offset=$PHI_OFFSET phi_beta=$PHI_BETA"
echo "           max_goal_offset=$MAX_GOAL_OFFSET p_random_goal=$P_RANDOM_GOAL wd=$WEIGHT_DECAY"
echo " eval    : N_EVALS=$N_EVALS w_qm=$W_QM w_l2=$W_L2 sweep=$DO_SWEEP"
echo "================================================================"
die () { echo ""; echo "!! FATAL: $1"; echo "   (see $RUN/relaunch.log)"; exit 1; }

# ---- 1. data ----------------------------------------------------------------
if [[ ! -f "$DATASET_DIR/pusht_noise/train/states.pth" ]]; then
  echo "==> [1] downloading pusht_noise + stock ckpt"
  bash scripts/download_data.sh 2>&1 | tee "$RUN/1_download.log"
fi
[[ -f "$DATASET_DIR/pusht_noise/train/states.pth" ]] || die "dataset missing after download"

# ---- 2. cache model-step latents -------------------------------------------
if [[ "$FORCE_CACHE" == 1 || ! -f "$CACHE/train/meta.json" ]]; then
  echo "==> [2a] caching TRAIN latents (n_rollout=$N_TRAIN, max_gb=$MAX_GB)"
  python scripts/cache_qm_latents.py --splits train --n_rollout "$N_TRAIN" --max_gb "$MAX_GB" \
      2>&1 | tee "$RUN/2a_cache_train.log"
fi
[[ -f "$CACHE/train/meta.json" ]] || die "train cache missing (disk guard tripped? see 2a_cache_train.log -- lower N_TRAIN or raise MAX_GB)"
if [[ "$FORCE_CACHE" == 1 || ! -f "$CACHE/val/meta.json" ]]; then
  echo "==> [2b] caching VAL latents (n_rollout=$N_VAL)"
  python scripts/cache_qm_latents.py --splits val --n_rollout "$N_VAL" --max_gb "$MAX_GB" \
      2>&1 | tee "$RUN/2b_cache_val.log"
fi
[[ -f "$CACHE/val/meta.json" ]] || die "val cache missing"

# ---- 3. train (calibrated, with LIVE val monitor) --------------------------
if [[ "$FORCE_TRAIN" == 1 || ! -f "$OUT/qm_head.pth" ]]; then
  echo "==> [3] training $TAG  (watch the [val] lines: val/train d_trans should stay ~1x)"
  python scripts/train_quasimetric.py --out "$OUT" --steps "$STEPS" \
      --phi_offset "$PHI_OFFSET" --phi_beta "$PHI_BETA" \
      --max_goal_offset "$MAX_GOAL_OFFSET" --p_random_goal "$P_RANDOM_GOAL" \
      --weight_decay "$WEIGHT_DECAY" --num_workers "$NUM_WORKERS" \
      2>&1 | tee "$RUN/3_train.log"
fi
[[ -f "$OUT/qm_head.pth" ]] || die "training produced no head (see 3_train.log)"

# Prefer the best-generalizing checkpoint the val monitor saved.
QM_CKPT="$OUT/qm_head_bestval.pth"; [[ -f "$QM_CKPT" ]] || QM_CKPT="$OUT/qm_head.pth"
echo "==> using head: $QM_CKPT"

# ---- 4. validation gates ----------------------------------------------------
echo "==> [4] validation gates on held-out val"
VALOUT="qm_outputs/validate_$TAG"
python analysis/validate_quasimetric.py --qm_ckpt "$QM_CKPT" \
    --cache_dir "$CACHE" --split val --out "$VALOUT" 2>&1 | tee "$RUN/4_validate.log"

# ---- 5. three-way eval (+ optional sweep) -----------------------------------
if [[ "$DO_SWEEP" == 1 ]]; then
  echo "==> [5a] w_qm:w_l2 knee sweep (n=10, NEW only)"
  QM_CKPT="$QM_CKPT" N_EVALS=10 STAMP="${TAG}_${STAMP}_sweep" bash analysis/run_qm_eval.sh sweep \
      2>&1 | tee "$RUN/5a_sweep.log"
fi
echo "==> [5b] headline 3-way (n=$N_EVALS): FLOOR / NEW / CEILING"
QM_CKPT="$QM_CKPT" N_EVALS="$N_EVALS" W_QM="$W_QM" W_L2="$W_L2" STAMP="${TAG}_${STAMP}" \
    bash analysis/run_qm_eval.sh all 2>&1 | tee "$RUN/5b_eval.log"

# ---- SUMMARY (tail of the master log = the decision data) -------------------
echo ""
echo "======================= RELAUNCH SUMMARY ($TAG) ======================="
echo "-- validation gates (must PASS (a)+(c) for NEW to have a chance) --"
grep -E "^\(a\)|^\(b\)|^\(c\)" "$RUN/4_validate.log" 2>/dev/null || echo "  (no gate lines -- check 4_validate.log)"
echo "-- three-way success rate (genuine held-out single-T goals) --"
grep -E "FLOOR|NEW|CEILING|SR=" "$RUN/5b_eval.log" 2>/dev/null | grep -E "SR=" || echo "  (no SR -- check 5b_eval.log)"
echo ""
echo "DECISION: does NEW close most of the FLOOR(~0.80)->CEILING(~0.97) gap?"
echo "All logs: $RUN/   head: $QM_CKPT   gates/plots: $VALOUT/"
echo "======================================================================="
