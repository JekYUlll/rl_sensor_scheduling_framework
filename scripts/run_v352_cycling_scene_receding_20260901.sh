#!/usr/bin/env bash
set -euo pipefail

# Label-free privileged structural diagnostic for the V352 cycling scene.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/.conda/envs/darts/bin/python}"
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local slot="$1" seed="$2"
  (
    export CUDA_VISIBLE_DEVICES="$((slot + GPU_OFFSET))"
    "$PY" scripts/99_v32_receding_upper.py \
      --run-dir "reports/v352_cycling_scene_control_seed${seed}_b1p75_20260822" \
      --output-subdir receding_oracle_l6_scene_gate \
      --receding-oracle-lookahead-steps 6 \
      --partition final_test --device cuda
  ) >"logs/v352_cycling_scene_receding_seed${seed}.log" 2>&1
}

mkdir -p logs
run_one 0 6891 & p1=$!
run_one 1 6892 & p2=$!
wait "$p1" "$p2"
