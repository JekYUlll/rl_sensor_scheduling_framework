#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_IDS="${GPU_IDS:-0 1}"
read -r -a gpu_ids <<< "$GPU_IDS"
variants=(ep80 ep200)
epochs=(80 200)
pids=()

for idx in "${!variants[@]}"; do
  variant="${variants[$idx]}"
  gpu="${gpu_ids[$((idx % ${#gpu_ids[@]}))]}"
  log_dir="logs/v190_v170_hard_teacher_pretrain_sweep"
  mkdir -p "$log_dir"
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
    SEEDS_OVERRIDE=1701 \
    GPU_IDS=0 \
    RUN_PREFIX_OVERRIDE="v190_hard_${variant}" \
    LOG_DIR_OVERRIDE="$log_dir/$variant" \
    BC_PRETRAIN_TARGET_MODE_OVERRIDE=hard \
    BC_PRETRAIN_EPOCHS_OVERRIDE="${epochs[$idx]}" \
    TOTAL_TIMESTEPS_OVERRIDE=0 \
    FORECAST_VALUE_AUX_LOSS_OVERRIDE=mse \
    FORECAST_VALUE_RANKING_COEF=0 \
    bash scripts/run_v174_v170_mse_value_utility_pilot_20260827.sh
  ) >"$log_dir/${variant}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
