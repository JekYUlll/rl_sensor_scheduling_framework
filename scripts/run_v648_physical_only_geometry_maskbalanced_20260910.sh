#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
RUN_ROOT="${V648_ASSET_ROOT:-reports/v648_physical_only_assets_maskbalanced_20260910}"
OUT_ROOT="${V648_GEOMETRY_ROOT:-reports/v648_physical_only_geometry_maskbalanced_20260910}"
mkdir -p "$OUT_ROOT"
declare -A STATE_STARTS=(
  [7401]="1000 1211 7717 82600"
  [7402]="1000 1087 7958 82600"
  [7403]="1000 1198 47626 82600"
  [7404]="1000 1203 7693 82600"
)
for seed in 7401 7402 7403 7404; do
  read -r s1 s2 s3 s4 <<< "${STATE_STARTS[$seed]}"
  "$PYTHON_BIN" scripts/109_v32_audit_subset_forecast_geometry.py \
    --run-dir "$RUN_ROOT/seed${seed}" --out-dir "$OUT_ROOT/seed${seed}" \
    --steps 256 --max-rollouts 4 \
    --start-index "$s1" --start-index "$s2" --start-index "$s3" --start-index "$s4" \
    --steady-budget 10.0 --startup-budget 10.0 --dynamic-resource-budget 55.0 \
    --oracle-loss-clip 1000000000 --torch-threads 1 \
    > "$OUT_ROOT/seed${seed}.log" 2>&1
done
