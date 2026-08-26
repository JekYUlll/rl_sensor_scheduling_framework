#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-1601 1602 1603 1604 1605}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4}"
pids=()
mkdir -p logs/v159_quality_validation_headroom

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  (
    export CUDA_VISIBLE_DEVICES="${GPUS[$((index % ${#GPUS[@]}))]}"
    "$PY" scripts/99_v32_receding_upper.py \
      --run-dir "reports/v152_channel_quality_scene_dev_seed${seed}_b1p75_20260822" \
      --output-subdir receding_oracle_l8_validation_gate \
      --partition validation \
      --device cuda
  ) >"logs/v159_quality_validation_headroom/seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
