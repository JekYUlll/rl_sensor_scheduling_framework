#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Final same-objective density test. No reward, scene, state, action, constraint,
# or privileged-information boundary changes relative to V141.
SEEDS_OVERRIDE="1505" \
GPU_IDS="0" \
RUN_PREFIX_OVERRIDE=v142_dense_value_regression_seed1505 \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.25 \
FORECAST_VALUE_AUX_COEF_OVERRIDE=1.0 \
FORECAST_VALUE_AUX_STRIDE_OVERRIDE=4 \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=mse \
ENT_COEF_OVERRIDE=0.005 \
bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh
