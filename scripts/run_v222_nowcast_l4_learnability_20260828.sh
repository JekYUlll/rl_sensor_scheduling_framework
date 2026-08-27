#!/usr/bin/env bash
set -euo pipefail

# Offline information diagnostic for V221.  Exact receding labels are produced
# only on policy-training and validation partitions; final traces are held out
# for the learnability report and never feed policy training.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/.conda/envs/darts/bin/python}"
RUN_PREFIX="${RUN_PREFIX:-v221_nowcast_normalized_dev}"
RUN_SUFFIX="${RUN_SUFFIX:-b1p85_20260822}"
OUT_ROOT="${OUT_ROOT:-reports/aggregate/v222_nowcast_l4_learnability_20260828}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-2411 2412 2413 2414 2415}"

mkdir -p "$OUT_ROOT" logs
pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  run_dir="reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}"
  (
    export CUDA_VISIBLE_DEVICES="$idx"
    "$PYTHON" scripts/99_v32_receding_upper.py \
      --run-dir "$run_dir" --partition rl_train \
      --output-subdir receding_oracle_l4_rl_train_trace \
      --receding-oracle-lookahead-steps 4 --device cuda
    "$PYTHON" scripts/99_v32_receding_upper.py \
      --run-dir "$run_dir" --partition validation \
      --output-subdir receding_oracle_l4_validation_trace \
      --receding-oracle-lookahead-steps 4 --device cuda
    "$PYTHON" scripts/99_v32_receding_upper.py \
      --run-dir "$run_dir" --partition final_test \
      --output-subdir receding_oracle_l4_final_trace \
      --receding-oracle-lookahead-steps 4 --device cuda
  ) >"logs/v222_l4_learnability_seed${seed}.log" 2>&1 &
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
  --run-dirs "${run_dirs[@]}" \
  --trace-prefix receding_oracle_l4 \
  --final-trace-subdir receding_oracle_l4_final_trace \
  --output-dir "$OUT_ROOT/learnability"
