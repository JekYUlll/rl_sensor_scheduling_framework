#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Frozen completion of V127/V128 on the remaining development scenes.
SEEDS_OVERRIDE="1304 1305" \
GPU_IDS="0 1" \
RUN_PREFIX_OVERRIDE=v129_soft_onpolicy_forecast_value \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=soft_ce \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=1.0 \
bash scripts/run_v125_dense_onpolicy_forecast_value_pilot_20260823.sh
