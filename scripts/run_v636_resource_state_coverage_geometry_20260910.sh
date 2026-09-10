#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
RUN_ROOT="${V636_ASSET_ROOT:-reports/v634_nowcast_operating_assets_b2p15_20260910}"
OUT_ROOT="${V636_GEOMETRY_ROOT:-reports/v636_resource_state_coverage_geometry_b2p15_20260910}"
cd "$ROOT"
mkdir -p "$OUT_ROOT"

# Starts were selected from the resource trace before reading any forecast
# losses: one common early reference, the first (0,0) heater state, the first
# (1,0) state where the laser channel can be feasible, and the existing late
# reference.  Seed 7179 reaches (1,0) later than the other seeds.
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
    --steady-budget 2.15 --startup-budget 2.60 --dynamic-resource-budget 2.15 \
    --oracle-loss-clip 1000000000 --torch-threads 1 \
    > "$OUT_ROOT/seed${seed}.log" 2>&1
done
