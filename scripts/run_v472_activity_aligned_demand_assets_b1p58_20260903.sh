#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUN_PREFIX_OVERRIDE=v472_activity_aligned_demand_assets_b1p58
export LOG_PREFIX_OVERRIDE=v472_activity_aligned_demand_assets_b1p58
export SOURCE_TRUTH_PREFIX_OVERRIDE=v470_activity_aligned_demand_truth
export SCENE_SEEDS_OVERRIDE="7091 7092"
export POLICY_SEEDS_OVERRIDE="9121 9122"
export GPU_OFFSET="${GPU_OFFSET:-1}"
export BUDGET_OVERRIDE=1.58 STARTUP_BUDGET_OVERRIDE=2.15
export BUDGET_LABEL_OVERRIDE=b1p58
export FORECAST_VALUE_HORIZON_PERSISTENT_LATENT=1
export FORECAST_VALUE_SPECIALIST_RESILIENT_QUALITY=1
export FORECAST_VALUE_ACTIVITY_ALIGNED_TRANSPORT_DEMAND=1

status="logs/v472_activity_aligned_demand_assets_b1p58_20260903.exit"
rm -f "$status"
bash scripts/run_v446_residence_forecast_value_assets_20260903.sh
rc=$?
printf '%s\n' "$rc" >"$status"
exit "$rc"
