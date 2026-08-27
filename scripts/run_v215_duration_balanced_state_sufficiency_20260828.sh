#!/usr/bin/env bash
set -euo pipefail

# Exploratory diagnostic only. Receding-oracle labels are generated exclusively
# on the policy-training partition. They train read-only alert-only/full-state
# probes and a trace-distilled policy; final traces are evaluation labels only.
# None of these artifacts define the primary PD-PPO method.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/.conda/envs/darts/bin/python}"
RUN_PREFIX="${RUN_PREFIX:-v213_duration_balanced_scene_dev}"
RUN_SUFFIX="${RUN_SUFFIX:-b1p85_20260822}"
OUT_ROOT="${OUT_ROOT:-reports/aggregate/v215_duration_balanced_state_sufficiency_20260828}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-2301 2302 2303 2304 2305}"

mkdir -p "$OUT_ROOT" logs
pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  run_dir="reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}"
  (
    export CUDA_VISIBLE_DEVICES="$idx"
    "$PYTHON" scripts/99_v32_receding_upper.py \
      --run-dir "$run_dir" --partition rl_train \
      --output-subdir receding_oracle_l8_rl_train_trace --device cuda
    "$PYTHON" scripts/99_v32_receding_upper.py \
      --run-dir "$run_dir" --partition validation \
      --output-subdir receding_oracle_l8_validation_trace --device cuda
    "$PYTHON" scripts/99_v32_receding_upper.py \
      --run-dir "$run_dir" --partition final_test \
      --output-subdir receding_oracle_l8_final_trace --device cuda
  ) >"logs/v215_state_sufficiency_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

run_dirs=()
for seed in "${SEEDS[@]}"; do
  run_dirs+=("reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}")
done
"$PYTHON" scripts/102_v32_receding_trace_learnability.py \
  --run-dirs "${run_dirs[@]}" --output-dir "$OUT_ROOT/learnability"
"$PYTHON" scripts/81_v31_framework_baseline_supplements.py \
  --run-glob "reports/${RUN_PREFIX}_seed*_${RUN_SUFFIX}" \
  --out-root "$OUT_ROOT/trace_distilled_replay" \
  --oracle-device cuda --policies trace_distilled
