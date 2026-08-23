#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# One-factor V125 control: retain the dense on-policy labels and replace only
# raw-logit MSE with a masked soft categorical target induced by forecast value.
RUN_PREFIX_OVERRIDE=v127_soft_onpolicy_forecast_value_pilot \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=soft_ce \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=1.0 \
bash scripts/run_v125_dense_onpolicy_forecast_value_pilot_20260823.sh
