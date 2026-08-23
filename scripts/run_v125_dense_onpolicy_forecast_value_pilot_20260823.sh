#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Bounded strength check after V124 showed directional validation improvement
# with a label rate of only 1.6 percent. All non-auxiliary settings are frozen.
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v125_dense_onpolicy_forecast_value_pilot}" \
FORECAST_VALUE_AUX_COEF_OVERRIDE=0.5 \
FORECAST_VALUE_AUX_STRIDE_OVERRIDE=16 \
bash scripts/run_v124_onpolicy_forecast_value_pilot_20260823.sh
