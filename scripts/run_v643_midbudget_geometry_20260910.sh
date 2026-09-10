#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
RUN_ROOT="${V643_ASSET_ROOT:-reports/v641_radiometer_signal_assets_b2p15_20260910}"
OUT_ROOT="${V643_GEOMETRY_ROOT:-reports/v643_midbudget_geometry_b1p98_20260910}"
cd "$ROOT"
mkdir -p "$OUT_ROOT"

# B=1.98 is fixed from the observed subset-cost breakpoint table.  It admits
# selected two-specialist masks (including radiometer+laser and met+FC4) while
# excluding the wider union. Starts remain selected from resource states only.
declare -A STATE_STARTS=(
  [7177]="1000 1211 7717 82600"
  [7178]="1000 1087 7958 82600"
  [7179]="1000 1198 47626 82600"
  [7180]="1000 1203 7693 82600"
)
for seed in 7177 7178 7179 7180; do
  read -r s1 s2 s3 s4 <<< "${STATE_STARTS[$seed]}"
  "$PYTHON_BIN" scripts/109_v32_audit_subset_forecast_geometry.py \
    --run-dir "$RUN_ROOT/seed${seed}" --out-dir "$OUT_ROOT/seed${seed}" \
    --steps 256 --max-rollouts 4 \
    --start-index "$s1" --start-index "$s2" --start-index "$s3" --start-index "$s4" \
    --steady-budget 1.98 --startup-budget 2.40 --dynamic-resource-budget 1.98 \
    --oracle-loss-clip 1000000000 --torch-threads 1 \
    > "$OUT_ROOT/seed${seed}.log" 2>&1
done
