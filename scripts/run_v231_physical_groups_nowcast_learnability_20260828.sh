#!/usr/bin/env bash
set -euo pipefail

# Read-only learnability diagnostic for V229. Receding labels are produced on
# chronological policy-training and validation partitions; a regressor fitted
# on those online-state traces is evaluated only on held-out final traces.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2481 2482 2483 2484 2485}"
export RUN_PREFIX="${RUN_PREFIX:-v229_physical_groups_nowcast_ppo_dev}"
export RUN_SUFFIX="${RUN_SUFFIX:-b1p85_20260822}"
export OUT_ROOT="${OUT_ROOT:-reports/aggregate/v231_physical_groups_nowcast_learnability_20260828}"

PYTHON="${PYTHON:-$HOME/.conda/envs/darts/bin/python}"
mkdir -p "$OUT_ROOT" logs
pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  run_dir="reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}"
  (
    export CUDA_VISIBLE_DEVICES="$idx"
    "$PYTHON" scripts/99_v32_receding_upper.py --run-dir "$run_dir" \
      --partition rl_train --output-subdir receding_oracle_l8_rl_train_trace \
      --receding-oracle-lookahead-steps 8 --device cuda
    "$PYTHON" scripts/99_v32_receding_upper.py --run-dir "$run_dir" \
      --partition validation --output-subdir receding_oracle_l8_validation_trace \
      --receding-oracle-lookahead-steps 8 --device cuda
    "$PYTHON" scripts/99_v32_receding_upper.py --run-dir "$run_dir" \
      --partition final_test --output-subdir receding_oracle_l8_final_trace \
      --receding-oracle-lookahead-steps 8 --device cuda
  ) >"logs/v231_l8_learnability_seed${seed}.log" 2>&1 &
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
  --run-dirs "${run_dirs[@]}" --trace-prefix receding_oracle_l8 \
  --final-trace-subdir receding_oracle_l8_final_trace \
  --output-dir "$OUT_ROOT/learnability"
