#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
SEEDS=( $(seq 1401 1424) )
OUT=reports/aggregate/v135_fresh_strong_references_20260826
mkdir -p "$OUT" logs/v135_fresh_strong_references

for ((start=0; start<${#SEEDS[@]}; start+=6)); do
  pids=()
  for ((slot=0; slot<6 && start+slot<${#SEEDS[@]}; slot++)); do
    seed="${SEEDS[$((start+slot))]}"
    (
      export CUDA_VISIBLE_DEVICES="$slot"
      env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" \
        scripts/81_v31_framework_baseline_supplements.py \
        --run-glob "reports/v120_full_intensity_context_gate_dev_seed${seed}_b1p75_20260822" \
        --seeds "$seed" --out-root "$OUT" --router-eval-dir . \
        --replay-dir __none__ --oracle-device cuda \
        --policies context_bandit forecast_greedy \
        --context-thresholds 0.5 --context-action-source replay_calibrated \
        --greedy-max-steps 0 --no-aggregate
    ) >"logs/v135_fresh_strong_references/seed${seed}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" \
  scripts/81_v31_framework_baseline_supplements.py \
  --run-glob 'reports/v120_full_intensity_context_gate_dev_seed*_b1p75_20260822' \
  --seeds "${SEEDS[@]}" --out-root "$OUT" --router-eval-dir . \
  --replay-dir __none__ --oracle-device cpu \
  --policies context_bandit forecast_greedy \
  --context-thresholds 0.5 --context-action-source replay_calibrated \
  --greedy-max-steps 0 --reuse-existing-seed-metrics
