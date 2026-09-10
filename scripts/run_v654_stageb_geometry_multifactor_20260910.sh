#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
RUN_ROOT="${V654_RUN_ROOT:-reports/v651_stageb_maskbalanced_assets_20260910}"
OUT_ROOT="${V654_OUT_ROOT:-reports/v654_stageb_geometry_multifactor_20260910}"
mkdir -p "$OUT_ROOT"
for seed in 7231 7232 7233 7234; do
  out="$OUT_ROOT/seed${seed}"
  mkdir -p "$out"
  "$PYTHON_BIN" scripts/109_v32_audit_subset_forecast_geometry.py \
    --run-dir "$RUN_ROOT/seed${seed}" --out-dir "$out/train" \
    --steps 256 --max-rollouts 3 \
    --start-index 77452 --start-index 78907 --start-index 80344 \
    --steady-budget 20.0 --startup-budget 25.0 --dynamic-resource-budget 20.0 \
    --oracle-loss-clip 1000000000 --torch-threads 1 > "$out/train.log" 2>&1
  "$PYTHON_BIN" scripts/109_v32_audit_subset_forecast_geometry.py \
    --run-dir "$RUN_ROOT/seed${seed}" --out-dir "$out/test" \
    --steps 256 --max-rollouts 1 --start-index 82600 \
    --steady-budget 20.0 --startup-budget 25.0 --dynamic-resource-budget 20.0 \
    --oracle-loss-clip 1000000000 --torch-threads 1 > "$out/test.log" 2>&1
done
