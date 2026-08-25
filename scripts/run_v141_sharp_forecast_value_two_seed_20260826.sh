#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Bounded correction after V139 showed diffuse value targets and deterministic
# action collapse. Only target sharpness and entropy regularization change.
SEEDS_OVERRIDE="1502 1505" \
GPU_IDS="0 1" \
RUN_PREFIX_OVERRIDE=v141_sharp_forecast_value_two_seed \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.25 \
ENT_COEF_OVERRIDE=0.005 \
bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh
