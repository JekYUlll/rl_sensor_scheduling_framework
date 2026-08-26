#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# V144 matched settings with only the forecast-head candidate representation
# changed from factorized subset embeddings to independent candidate outputs.
SEEDS_OVERRIDE="1505" \
GPU_IDS="0" \
RUN_PREFIX_OVERRIDE=v145_independent_forecast_head_seed1505 \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.25 \
FORECAST_VALUE_AUX_COEF_OVERRIDE=1.0 \
FORECAST_VALUE_AUX_STRIDE_OVERRIDE=4 \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=mse \
ENT_COEF_OVERRIDE=0.005 \
FORECAST_VALUE_HEAD=1 \
FORECAST_VALUE_HEAD_SCALE=1.0 \
FORECAST_VALUE_HEAD_HIDDEN_DIM=128 \
FORECAST_VALUE_HEAD_MODE=independent \
bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh
