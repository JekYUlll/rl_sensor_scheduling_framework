#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
RUN_ROOT="${V662_RUN_ROOT:-reports/v659_observable_specialist_mode_assets_20260910}"
OUT_ROOT="${V662_OUT_ROOT:-reports/v662_observable_specialist_mode_geometry_testgrid_20260910}"
mkdir -p "$OUT_ROOT"
for seed in 7231 7232 7233 7234; do
  for start in 82600 83900 85200 86500 87800; do
    out="$OUT_ROOT/seed${seed}/test_${start}"
    mkdir -p "$out"
    "$PYTHON_BIN" scripts/109_v32_audit_subset_forecast_geometry.py \
      --run-dir "$RUN_ROOT/seed${seed}" --out-dir "$out" \
      --steps 256 --max-rollouts 1 --start-index "$start" \
      --steady-budget 20.0 --startup-budget 25.0 --dynamic-resource-budget 5.0 \
      --oracle-loss-clip 1000000000 --torch-threads 1 > "$out/audit.log" 2>&1
  done
done
