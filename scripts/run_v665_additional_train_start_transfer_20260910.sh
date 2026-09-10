#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
ASSET_ROOT="${V665_ASSET_ROOT:-reports/v659_observable_specialist_mode_assets_20260910}"
GEOM_ROOT="${V665_GEOM_ROOT:-reports/v665_additional_train_start_geometry_20260910}"
OBS_ROOT="${V665_OBS_ROOT:-reports/v665_additional_train_start_observations_20260910}"
for seed in 7231 7232 7233 7234; do
  geom="$GEOM_ROOT/seed${seed}"
  mkdir -p "$geom" "$OBS_ROOT/seed${seed}"
  "$PYTHON_BIN" scripts/109_v32_audit_subset_forecast_geometry.py \
    --run-dir "$ASSET_ROOT/seed${seed}" --out-dir "$geom" \
    --steps 256 --max-rollouts 1 --start-index 81789 \
    --steady-budget 20.0 --startup-budget 25.0 --dynamic-resource-budget 5.0 \
    --oracle-loss-clip 1000000000 --torch-threads 1 > "$geom/audit.log" 2>&1
  "$PYTHON_BIN" scripts/168_dump_online_observation_probe.py \
    --run-dir "$ASSET_ROOT/seed${seed}" --out-dir "$OBS_ROOT/seed${seed}" \
    --steps 256 --dynamic-resource-budget 5.0 --dynamic-resource-fixed-power 0.4104 \
    --start-index 81789 > "$OBS_ROOT/seed${seed}.log" 2>&1
done
