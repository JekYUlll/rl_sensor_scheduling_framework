#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
out="reports/analysis/v474_activity_aligned_demand_high_precision_observability_20260903"
state_out="reports/analysis/v475_activity_aligned_demand_state_transfer_20260903"
log="logs/v474_v475_activity_aligned_observability_b1p58_20260903.log"
status="logs/v474_v475_activity_aligned_observability_b1p58_20260903.exit"
runs=(
  reports/v472_activity_aligned_demand_assets_b1p58_seed7091_b1p58_20260822
  reports/v472_activity_aligned_demand_assets_b1p58_seed7092_b1p58_20260822
)
geometry="reports/analysis/v473r2_activity_aligned_demand_geometry_b1p58_20260903/geometry/subset_forecast_geometry_summary.json"
hindsight="reports/analysis/v473_activity_aligned_demand_geometry_b1p58_20260903/hindsight_h24/dwell_hindsight_summary.json"
mkdir -p logs
rm -f "$status"
{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  py="${PYTHON_BIN:-$HOME/.conda/envs/darts/bin/python}"
  "$py" - "$geometry" "$hindsight" <<'PY' || exit $?
import json, sys
geometry = json.load(open(sys.argv[1]))
hindsight = json.load(open(sys.argv[2]))
if {int(row["seed"]) for row in geometry} != {7091, 7092}:
    raise SystemExit("V473r2 geometry seeds are incomplete")
for row in geometry:
    if row["near_optimal_intersections"]["0.01"]:
        raise SystemExit("event-domain epsilon-0.01 intersection is not empty")
    if row["operating_near_optimal_intersections"]["0.01"]:
        raise SystemExit("operating-domain epsilon-0.01 intersection is not empty")
if {int(row["seed"]) for row in hindsight} != {7091, 7092}:
    raise SystemExit("V473 hindsight seeds are incomplete")
for row in hindsight:
    if float(row["static_minus_hindsight_loss"]) <= 0.0:
        raise SystemExit("common-dwell opportunity is not positive")
    if int(row["hindsight_warmup_abort_count"]) != 0:
        raise SystemExit("common-dwell controller has warm-up aborts")
PY
  test ! -e "$out" || exit 2
  test ! -e "$state_out" || exit 2
  "$py" scripts/118_v32_audit_expected_cost_observability.py \
    --run-dir "${runs[0]}" --run-dir "${runs[1]}" \
    --out-dir "$out" --train-rollouts 12 --test-rollouts 6 --steps 128 \
    --lookahead-steps 24 --replicas 8 --epochs 100 \
    --steady-budget 1.58 --startup-budget 2.15 --torch-threads 1 \
    --static-comparator-source validation_ledger --save-datasets || exit $?
  "$py" scripts/120_v32_audit_state_bin_transfer.py \
    --dataset "$out/expected_cost_observability_datasets.npz" \
    --run-dir "${runs[0]}" --run-dir "${runs[1]}" \
    --out-dir "$state_out"
  rc=$?
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "$rc"
  printf '%s\n' "$rc" >"$status"
  exit "$rc"
} >>"$log" 2>&1
