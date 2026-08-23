#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Final bracketed target-temperature check after V129 and V130.
SEEDS_OVERRIDE="1304 1305" \
GPU_IDS="0 1" \
RUN_PREFIX_OVERRIDE=v131_soft_onpolicy_temp075 \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=soft_ce \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.75 \
bash scripts/run_v125_dense_onpolicy_forecast_value_pilot_20260823.sh
