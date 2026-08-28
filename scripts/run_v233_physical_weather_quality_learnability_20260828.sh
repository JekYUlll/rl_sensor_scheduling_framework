#!/usr/bin/env bash
set -euo pipefail

# Read-only information gate for V232. Candidate losses are generated only on
# policy-training and validation partitions. The deployed diagnostic policy is
# then replayed solely on each frozen final partition.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2501 2502 2503 2504 2505}"
export RUN_PREFIX="${RUN_PREFIX:-v232_physical_weather_quality_dev}"
export RUN_SUFFIX="${RUN_SUFFIX:-b1p85_20260822}"
export OUT_ROOT="${OUT_ROOT:-reports/aggregate/v233_physical_weather_quality_learnability_20260828}"
read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
PYTHON="${PYTHON:-$HOME/.conda/envs/darts/bin/python}"
mkdir -p "$OUT_ROOT" logs

pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  run="reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}"
  (
    export CUDA_VISIBLE_DEVICES="$idx"
    "$PYTHON" scripts/99_v32_receding_upper.py --run-dir "$run" --partition rl_train \
      --output-subdir receding_oracle_l8_rl_train_trace --receding-oracle-lookahead-steps 8 --device cuda
    "$PYTHON" scripts/99_v32_receding_upper.py --run-dir "$run" --partition validation \
      --output-subdir receding_oracle_l8_validation_trace --receding-oracle-lookahead-steps 8 --device cuda
  ) >"logs/v233_learnability_seed${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

run_dirs=()
for seed in "${SEEDS[@]}"; do run_dirs+=("reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}"); done
"$PYTHON" scripts/102_v32_receding_trace_learnability.py \
  --run-dirs "${run_dirs[@]}" --trace-prefix receding_oracle_l8 \
  --final-trace-subdir receding_oracle_l8_scene_gate --output-dir "$OUT_ROOT/learnability"

# Run the all-action trace regressor one seed at a time to avoid nested
# multi-core ExtraTrees oversubscription on the shared remote host.
for seed in "${SEEDS[@]}"; do
  "$PYTHON" scripts/81_v31_framework_baseline_supplements.py \
    --run-glob "reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}" --seeds "$seed" \
    --out-root "$OUT_ROOT/trace_distilled" --router-eval-dir . --replay-dir __none__ \
    --oracle-device cuda --policies trace_distilled \
    --trace-training-subdir receding_oracle_l8_rl_train_trace --no-aggregate \
    >"logs/v233_trace_distilled_seed${seed}.log" 2>&1
done

exec "$PYTHON" scripts/108_v32_collect_trace_distilled_gate.py \
  --prefix "$RUN_PREFIX" --baseline-root "$OUT_ROOT/trace_distilled" \
  --out-dir "$OUT_ROOT/trace_distilled_summary" --seeds "${SEEDS[@]}"
