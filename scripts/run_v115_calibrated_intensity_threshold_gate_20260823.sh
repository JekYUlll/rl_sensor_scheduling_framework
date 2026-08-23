#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_GLOB="reports/v113_frequency_cost_intensity_context_gate_dev_seed*_b1p75_20260822"
CANDIDATE_ROOT="reports/aggregate/v115_intensity_threshold_candidates_20260823"
OUT_ROOT="reports/aggregate/v115_calibrated_intensity_context_gate_20260823"
mkdir -p "$CANDIDATE_ROOT" "$OUT_ROOT" logs

pids=()
for seed in 1301 1302 1303 1304 1305; do
  gpu=$((seed - 1301))
  (
    for high in 0.65 0.75 0.85; do
      tag="$(printf '%s' "$high" | tr . p)"
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/81_v31_framework_baseline_supplements.py \
        --run-glob "$RUN_GLOB" \
        --seeds "$seed" \
        --out-root "$CANDIDATE_ROOT/high_${tag}" \
        --policies context_bandit \
        --context-thresholds 0.5 \
        --context-high-threshold "$high" \
        --context-action-source intensity_replay_calibrated \
        --oracle-device cuda \
        --no-aggregate
    done
  ) > "logs/v115_intensity_threshold_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python scripts/100_v32_select_intensity_context_threshold.py \
  --candidate-glob "$CANDIDATE_ROOT/high_*/seed*/framework_baseline_metrics.csv" \
  --out-root "$OUT_ROOT"
