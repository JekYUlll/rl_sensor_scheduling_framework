#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
ASSET_ROOT="${V678_ASSET_ROOT:-reports/v675_split_affinity_quality_assets_b2p15_20260911}"
OUT_ROOT="${V678_OUT_ROOT:-reports/v678_binding_specialist_pair_geometry_b2p50_20260911}"
mkdir -p "$OUT_ROOT"
START_ARGS=(
  --start-index 82600 --start-index 83000
  --start-index 84800 --start-index 85500
)
for seed in 7177 7178 7179 7180; do
  "$PYTHON_BIN" scripts/109_v32_audit_subset_forecast_geometry.py \
    --run-dir "$ASSET_ROOT/seed${seed}" \
    --out-dir "$OUT_ROOT/seed${seed}" \
    --steps 256 --max-rollouts 4 "${START_ARGS[@]}" \
    --steady-budget 2.50 --startup-budget 2.50 \
    --dynamic-resource-budget 2.50 \
    --oracle-loss-clip 100 --torch-threads 1 \
    > "$OUT_ROOT/seed${seed}.log" 2>&1
done
