#!/usr/bin/env bash
# Stage-1 corpus-scale dynamics retrain (gated, kill-criterion-probed). See docs/CORPUS_SCALE_SCOPE.md.
# gen 3k traj (16 workers) -> cache -> R -> dynamics (ckpt every 8) -> D2 probe (n=100).
# Run on box: nohup bash run_stage1.sh > /workspace/stage1.log 2>&1 &
set -uo pipefail
cd /workspace/langtable_kit
LT=/workspace/envs/langtable/bin/python
DW=/workspace/envs/dino_wm/bin/python
TRAJ=/workspace/lt_traj_3k; CACHE=/workspace/lt_cache_3k; G2=/workspace/g2_3k; RO=/workspace/readout_3k
EPISODES=${1:-102}   # 30 x 102 = 3060 traj
NW=${2:-30}          # single-threaded workers (box has 32 cores); avoid BLAS/pybullet thread thrash

echo "===[STAGE1 GEN]=== $NW workers x $EPISODES eps (1 thread/worker)  $(date)"
rm -rf $TRAJ; mkdir -p $TRAJ
for k in $(seq 0 $((NW-1))); do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    $LT -u lt_dump_traj.py --episodes $EPISODES --seed $k --out_dir $TRAJ > $TRAJ/gen_w$k.log 2>&1 &
done
wait
NG=$(ls $TRAJ/w*_t*.npz 2>/dev/null | wc -l)
echo "gen done: $NG traj files  $(date)"
if [ "$NG" -lt 2000 ]; then echo "ABORT: too few traj ($NG)"; exit 1; fi

echo "===[STAGE1 CACHE]===  $(date)"
$DW -u lt_cache.py --traj_dir $TRAJ --out_dir $CACHE --frameskip 5 || { echo "ABORT cache"; exit 1; }

echo "===[STAGE1 READOUT R]===  $(date)"
$DW -u lt_readout.py --cache $CACHE --out $RO || { echo "ABORT readout"; exit 1; }

echo "===[STAGE1 DYNAMICS]=== epochs=40 iters=1200 ckpt_every=8  $(date)"
$DW -u lt_g2.py --cache $CACHE --out $G2 --epochs 40 --iters_per_epoch 1200 --ckpt_every 8 || { echo "ABORT train"; exit 1; }

echo "===[STAGE1 D2 PROBE n=100]===  $(date)"
$DW -u lt_relplan.py --d2only --cache $CACHE --model $G2/model.pth --readout $RO/R.pth --n_d2 100

echo "===[STAGE1 DONE]===  $(date)"
