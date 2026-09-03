#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
out="reports/analysis/v473_activity_aligned_demand_geometry_b1p58_20260903"
log="logs/v473_activity_aligned_demand_geometry_b1p58_20260903.log"
status="logs/v473_activity_aligned_demand_geometry_b1p58_20260903.exit"
runs=(
  reports/v472_activity_aligned_demand_assets_b1p58_seed7091_b1p58_20260822
  reports/v472_activity_aligned_demand_assets_b1p58_seed7092_b1p58_20260822
)
mkdir -p logs
rm -f "$status"
{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  for run in "${runs[@]}"; do
    test -s "$run/v2_tcn_oracle.pt" || exit 2
    test -s "$run/validation_static_candidates.csv" || exit 2
    test ! -e "$run/custom_ppo.pt" || exit 2
  done
  test ! -e "$out" || exit 2
  mkdir -p "$out"/{resource_geometry,geometry,hindsight_h24}
  py="${PYTHON_BIN:-$HOME/.conda/envs/darts/bin/python}"
  "$py" scripts/119_v32_audit_resource_geometry.py \
    --sensor-cfg configs/sensors/windblown_sensors_flexible_subset_v6_physical_channels.yaml \
    --steady-budget 1.58 --startup-budget 2.15 \
    --out-dir "$out/resource_geometry/audit" || exit $?
  "$py" scripts/109_v32_audit_subset_forecast_geometry.py \
    --run-dir "${runs[0]}" --run-dir "${runs[1]}" \
    --out-dir "$out/geometry" --steps 256 --max-rollouts 2 \
    --epsilon 0.01 --epsilon 0.05 --steady-budget 1.58 \
    --startup-budget 2.15 --torch-threads 1 || exit $?
  "$py" scripts/113_v32_audit_dwell_hindsight_opportunity.py \
    --run-dir "${runs[0]}" --run-dir "${runs[1]}" \
    --out-dir "$out/hindsight_h24" --steps 256 --max-rollouts 2 \
    --lookahead-steps 24 --steady-budget 1.58 \
    --startup-budget 2.15 --torch-threads 1
  rc=$?
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "$rc"
  printf '%s\n' "$rc" >"$status"
  exit "$rc"
} >>"$log" 2>&1
