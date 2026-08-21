#!/usr/bin/env bash
set -euo pipefail

source /data/conda_backup/miniconda3/etc/profile.d/conda.sh
conda activate darts

cd ~/_code/microclimate_demo/rl_sensor_scheduling_framework
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

BASE_OUT=reports/v2_forecast_eval_grid_dqn_d2_500k_nstep8_full
PPO_OUT=reports/v2_forecast_eval_grid_prior_kl1
TABLE_OUT=reports/v2_paper_tables_dqn_d2_500k_nstep8_full
ASSET_OUT=reports/v2_paper_assets_dqn_d2_500k_nstep8_full
mkdir -p "$BASE_OUT" "$TABLE_OUT" "$ASSET_OUT"

run_seed() {
  local seed="$1"
  local gpu="$2"
  echo "[D2 seed ${seed}] start on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/26_v2_grid_experiment.py \
    --policy dqn \
    --base-out-dir "$BASE_OUT" \
    --budgets 1.65 1.70 1.75 \
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
    > "$BASE_OUT/seed${seed}.log" 2>&1
  echo "[D2 seed ${seed}] done"
}

run_seed 41 0 &
run_seed 42 1 &
run_seed 43 2 &
wait

python scripts/27_v2_aggregate_results.py \
  --input-dirs "$PPO_OUT" "$BASE_OUT" \
  --output-dir "$TABLE_OUT"

python scripts/28_v2_build_paper_assets.py \
  --tables-dir "$TABLE_OUT" \
  --output-dir "$ASSET_OUT"

echo "D2 complete: $TABLE_OUT $ASSET_OUT"
