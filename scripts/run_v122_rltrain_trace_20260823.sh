#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/.conda/envs/darts/bin/python}"
RUN_PREFIX="${RUN_PREFIX:-v120_full_intensity_context_gate_dev}"
RUN_SUFFIX="${RUN_SUFFIX:-b1p75_20260822}"
SEEDS=(1301 1302 1303 1304 1305)

cd "$ROOT"
pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  run_dir="reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}"
  CUDA_VISIBLE_DEVICES="$index" "$PYTHON" scripts/99_v32_receding_upper.py \
    --run-dir "$run_dir" \
    --partition rl_train \
    --output-subdir receding_oracle_l8_rl_train_trace \
    --device cuda >"logs/v122_trace_seed${seed}_20260823.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "V122_EXIT=1"
  exit "$status"
fi

run_dirs=()
for seed in "${SEEDS[@]}"; do
  run_dirs+=("reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}")
done
"$PYTHON" scripts/102_v32_receding_trace_learnability.py \
  --run-dirs "${run_dirs[@]}" \
  --output-dir reports/aggregate/v122_rltrain_trace_learnability_20260823
echo "V122_EXIT=0"
