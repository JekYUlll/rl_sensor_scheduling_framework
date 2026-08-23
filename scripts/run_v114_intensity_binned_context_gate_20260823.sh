#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="reports/aggregate/v114_intensity_binned_context_gate_20260823"
RUN_GLOB="reports/v113_frequency_cost_intensity_context_gate_dev_seed*_b1p75_20260822"
mkdir -p "$OUT_ROOT" logs

pids=()
for seed in 1301 1302 1303 1304 1305; do
  CUDA_VISIBLE_DEVICES=$((seed - 1301)) python scripts/81_v31_framework_baseline_supplements.py \
    --run-glob "$RUN_GLOB" \
    --seeds "$seed" \
    --out-root "$OUT_ROOT" \
    --policies context_bandit \
    --context-thresholds 0.5 \
    --context-high-threshold 0.75 \
    --context-action-source intensity_replay_calibrated \
    --oracle-device cuda \
    --no-aggregate \
    > "logs/v114_intensity_context_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python scripts/81_v31_framework_baseline_supplements.py \
  --run-glob "$RUN_GLOB" \
  --out-root "$OUT_ROOT" \
  --policies context_bandit \
  --context-thresholds 0.5 \
  --context-high-threshold 0.75 \
  --context-action-source intensity_replay_calibrated \
  --reuse-existing-seed-metrics \
  --oracle-device cpu
