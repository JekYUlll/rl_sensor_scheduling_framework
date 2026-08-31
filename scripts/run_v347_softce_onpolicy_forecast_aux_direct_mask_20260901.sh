#!/usr/bin/env bash
set -euo pipefail

# V347 changes only the V346 on-policy candidate-value auxiliary from masked
# MSE to masked soft cross-entropy. The forecast-loss reward, action geometry,
# feasibility mask, and online information boundary remain unchanged.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_PREFIX_OVERRIDE="v347_softce_onpolicy_forecast_aux_direct_mask" \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=soft_ce \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=1.0 \
bash scripts/run_v346_onpolicy_forecast_aux_direct_mask_20260901.sh
