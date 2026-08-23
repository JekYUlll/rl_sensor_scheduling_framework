#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Frozen completion of temperature0.75 on the other development scenes.
SEEDS_OVERRIDE="1301 1302 1303" \
GPU_IDS="0 1 2" \
RUN_PREFIX_OVERRIDE=v132_soft_onpolicy_temp075 \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=soft_ce \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.75 \
bash scripts/run_v125_dense_onpolicy_forecast_value_pilot_20260823.sh
