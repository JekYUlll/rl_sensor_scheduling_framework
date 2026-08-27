#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
read -r -a gpu_ids <<< "$GPU_IDS"
variants=(rank025_ep20 rank025_ep80 rank100_ep20 rank100_ep80)
ranks=(0.25 0.25 1.0 1.0)
epochs=(20 80 20 80)
pids=()

for idx in "${!variants[@]}"; do
  variant="${variants[$idx]}"
  gpu="${gpu_ids[$((idx % ${#gpu_ids[@]}))]}"
  log_dir="logs/v189_v170_direct_value_pretrain_sweep"
  mkdir -p "$log_dir"
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
    SEEDS_OVERRIDE=1701 \
    GPU_IDS=0 \
    RUN_PREFIX_OVERRIDE="v189_${variant}" \
    LOG_DIR_OVERRIDE="$log_dir/$variant" \
    BC_PRETRAIN_EPOCHS_OVERRIDE="${epochs[$idx]}" \
    TOTAL_TIMESTEPS_OVERRIDE=0 \
    FORECAST_VALUE_AUX_LOSS_OVERRIDE=smooth_l1 \
    FORECAST_VALUE_RANKING_COEF="${ranks[$idx]}" \
    bash scripts/run_v174_v170_mse_value_utility_pilot_20260827.sh
  ) >"$log_dir/${variant}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
