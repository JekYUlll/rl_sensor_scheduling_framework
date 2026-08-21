#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/darts/bin/python}"
OUT_ROOT="${OUT_ROOT:-reports/aggregate/pdppo_framework_baselines_clean_24seed_20260718}"
LOG_ROOT="${LOG_ROOT:-logs/framework_baselines_clean_parallel_20260718}"
MAX_WORKERS="${MAX_WORKERS:-8}"
SEEDS=(117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140)

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_seed() {
  local seed="$1"
  env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    "$PYTHON_BIN" scripts/81_v31_framework_baseline_supplements.py \
      --run-glob "reports/v31_scenebal2_matched_reward_forecast_noexactevent_seed${seed}_h075forecastctrl_20260718cleanpilot" \
      --seeds "$seed" \
      --out-root "$OUT_ROOT" \
      --router-eval-dir . \
      --replay-dir __none__ \
      --oracle-device cpu \
      --policies context_bandit forecast_greedy event_label \
      --context-thresholds 0.5 \
      --greedy-max-steps 0 \
      --reuse-existing-seed-metrics \
      --no-aggregate \
      >"$LOG_ROOT/seed${seed}.log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  while (( $(jobs -pr | wc -l) >= MAX_WORKERS )); do
    wait -n
  done
  run_seed "$seed" &
done
wait

env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$PYTHON_BIN" scripts/81_v31_framework_baseline_supplements.py \
    --run-glob "reports/v31_scenebal2_matched_reward_forecast_noexactevent_seed*_h075forecastctrl_20260718cleanpilot" \
    --seeds "${SEEDS[@]}" \
    --out-root "$OUT_ROOT" \
    --router-eval-dir . \
    --replay-dir __none__ \
    --oracle-device cpu \
    --policies context_bandit forecast_greedy event_label \
    --context-thresholds 0.5 \
    --greedy-max-steps 0 \
    --reuse-existing-seed-metrics \
    >"$LOG_ROOT/aggregate.log" 2>&1

printf 'framework_baselines_complete out_root=%s\n' "$OUT_ROOT"
