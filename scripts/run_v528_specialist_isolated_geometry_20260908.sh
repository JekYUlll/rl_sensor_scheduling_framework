#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
OUT="${V528_OUT_ROOT:-reports/analysis/v528_specialist_isolated_geometry_20260908}"
RUN_ROOT="${V527_ASSET_ROOT:-reports/v527_specialist_isolated_assets_20260908}"
test ! -e "$OUT"
for seed in 7181 7182 7183 7184; do
  run="$RUN_ROOT/seed${seed}_b2p15"
  test -s "$run/v2_tcn_oracle.pt"
  test -s "$run/validation_static_candidates.csv"
  test ! -e "$run/custom_ppo.pt"
done
mkdir -p "$OUT"
exec "$PYTHON_BIN" scripts/109_v32_audit_subset_forecast_geometry.py \
  --run-dir "$RUN_ROOT/seed7181_b2p15" --run-dir "$RUN_ROOT/seed7182_b2p15" \
  --run-dir "$RUN_ROOT/seed7183_b2p15" --run-dir "$RUN_ROOT/seed7184_b2p15" \
  --out-dir "$OUT" --steps 256 --max-rollouts 2 \
  --epsilon 0.01 --epsilon 0.05 --steady-budget 2.15 \
  --startup-budget 2.60 --torch-threads 1
