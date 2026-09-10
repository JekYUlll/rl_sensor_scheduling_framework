#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
TRUTH_ROOT="${V681_TRUTH_ROOT:-reports/v527_observable_target_truth_20260909_r2}"
OUT_ROOT="${V681_OUT_ROOT:-reports/v681_observable_target_resource_truth_20260911}"
SENSOR_CONFIG="${V681_SENSOR_CONFIG:-configs/sensors/windblown_sensors_entity_effective_cost_v1.yaml}"
mkdir -p "$OUT_ROOT"
for seed in 7177 7178 7179 7180; do
  "$PYTHON_BIN" scripts/187_build_v527_observable_resource_trace.py \
    --truth "$TRUTH_ROOT/truth_seed${seed}.csv" \
    --output "$OUT_ROOT/resource_seed${seed}.csv" \
    > "$OUT_ROOT/resource_seed${seed}.log"
  "$PYTHON_BIN" scripts/188_audit_observable_resource_screen.py \
    --trace "$OUT_ROOT/resource_seed${seed}.csv" \
    --sensor-config "$SENSOR_CONFIG" \
    --output "$OUT_ROOT/resource_seed${seed}_screen.json" \
    > "$OUT_ROOT/resource_seed${seed}_screen.log"
done
