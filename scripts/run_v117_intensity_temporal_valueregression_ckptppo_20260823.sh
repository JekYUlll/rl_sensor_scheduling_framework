#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Replace V116's unstable hard argmin BC labels with standardized all-action
# forecast-value regression. Scene, online inputs, PPO objective, constraints,
# seeds, and validation checkpoint selection remain matched.
BC_PRETRAIN_TARGET_MODE_OVERRIDE=forecast_value_regression \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v117_intensity_temporal_valueregression_ckptppo}" \
bash scripts/run_v116_intensity_temporal_forecastbc_ckptppo_20260823.sh
