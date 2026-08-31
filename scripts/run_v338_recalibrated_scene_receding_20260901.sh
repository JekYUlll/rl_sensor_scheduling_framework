#!/usr/bin/env bash
set -euo pipefail

# Run the label-free privileged receding-horizon scene gate on the V338 control
# assets. This is a structural diagnostic, not a deployable policy result.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/.conda/envs/darts/bin/python}"
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local slot="$1" seed="$2"
  (
    export CUDA_VISIBLE_DEVICES="$((slot + GPU_OFFSET))"
    "$PY" scripts/99_v32_receding_upper.py \
      --run-dir "reports/v338_recalibrated_scene_control_seed${seed}_b1p75_20260822" \
      --output-subdir receding_oracle_l6_scene_gate \
      --receding-oracle-lookahead-steps 6 \
      --partition final_test --device cuda
  ) >"logs/v338_recalibrated_scene_receding_seed${seed}.log" 2>&1
}

mkdir -p logs
run_one 0 6871 & p1=$!
run_one 1 6872 & p2=$!
wait "$p1" "$p2"
