#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# One-factor target-sharpness check on the two V129 joint failures.
SEEDS_OVERRIDE="1304 1305" \
GPU_IDS="0 1" \
RUN_PREFIX_OVERRIDE=v130_soft_onpolicy_temp05 \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=soft_ce \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.5 \
bash scripts/run_v125_dense_onpolicy_forecast_value_pilot_20260823.sh
