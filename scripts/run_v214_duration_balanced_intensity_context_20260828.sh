#!/usr/bin/env bash
set -euo pipefail

# Zero-training online-identifiability diagnostic for frozen V213 scene assets.
# It replaces one action per alert type with validation-selected low/high alert
# actions. The 0.50/0.75 thresholds are fixed before final-partition replay.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/.conda/envs/darts/bin/python}"
OUT_ROOT="${OUT_ROOT:-reports/aggregate/v214_duration_balanced_intensity_context_20260828}"
RUN_GLOB="${RUN_GLOB:-reports/v213_duration_balanced_scene_dev_seed*_b1p85_20260822}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-2301 2302 2303 2304 2305}"

mkdir -p "$OUT_ROOT" logs
pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  (
    export CUDA_VISIBLE_DEVICES="$idx"
    env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PYTHON" \
      scripts/81_v31_framework_baseline_supplements.py \
      --run-glob "$RUN_GLOB" --seeds "$seed" --out-root "$OUT_ROOT" \
      --policies context_bandit --context-thresholds 0.5 \
      --context-high-threshold 0.75 \
      --context-action-source intensity_replay_calibrated \
      --oracle-device cuda --no-aggregate
  ) >"logs/v214_duration_intensity_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PYTHON" \
  scripts/81_v31_framework_baseline_supplements.py \
  --run-glob "$RUN_GLOB" --out-root "$OUT_ROOT" \
  --policies context_bandit --context-thresholds 0.5 \
  --context-high-threshold 0.75 \
  --context-action-source intensity_replay_calibrated \
  --reuse-existing-seed-metrics --oracle-device cpu
