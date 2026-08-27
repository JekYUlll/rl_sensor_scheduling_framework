#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1701}" \
GPU_IDS="${GPU_IDS:-0}" \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v188_v170_ranked_direct_value_pilot}" \
LOG_DIR_OVERRIDE="${LOG_DIR_OVERRIDE:-logs/v188_v170_ranked_direct_value_pilot}" \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=smooth_l1 \
FORECAST_VALUE_RANKING_COEF=0.1 \
bash scripts/run_v174_v170_mse_value_utility_pilot_20260827.sh
