#!/usr/bin/env bash
set -euo pipefail

source /data/conda_backup/miniconda3/etc/profile.d/conda.sh
conda activate darts

cd ~/_code/microclimate_demo/rl_sensor_scheduling_framework
export PYTHONPATH=src

BASE_OUT=reports/v2_forecast_eval_grid_dqn_d2_500k_nstep8_full
mkdir -p "$BASE_OUT"

run_seed() {
  local seed="$1"
  local gpu="$2"
  echo "[D2 budget1.70 seed ${seed}] start on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/26_v2_grid_experiment.py \
    --policy dqn \
    --base-out-dir "$BASE_OUT" \
    --budgets 1.70 \
    --seeds "$seed" \
    --truth-steps 8192 \
    --oracle-rollout-steps 2400 \
    --oracle-epochs 18 \
    --oracle-full-open-repeat 3 \
    --total-timesteps 500000 \
    --learning-rate 0.0001 \
    --batch-size 64 \
    --learning-starts 2000 \
    --train-freq 4 \
    --gradient-steps 1 \
    --target-update-interval 1000 \
    --replay-size 50000 \
    --n-step-return 8 \
    --dqn-max-candidate-warmup -1 \
    --eval-steps 1024 \
    --eval-rollouts 6 \
    --eval-event-fraction 0.67 \
    --device cuda \
    --oracle-device cuda \
    --oracle-inference-device cuda \
    --startup-peak-budget 3.2 \
    > "$BASE_OUT/budget1p70_seed${seed}_fast.log" 2>&1
  echo "[D2 budget1.70 seed ${seed}] done"
}

run_seed 41 3 &
run_seed 42 4 &
run_seed 43 5 &
wait

echo "D2 budget=1.70 fast branch complete."
