#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1701}" \
GPU_IDS="${GPU_IDS:-0}" \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v186_v170_ranked_mask_value_pilot}" \
LOG_DIR_OVERRIDE="${LOG_DIR_OVERRIDE:-logs/v186_v170_ranked_mask_value_pilot}" \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=smooth_l1 \
FORECAST_VALUE_RANKING_COEF=0.1 \
bash scripts/run_v185_v170_mask_structured_value_pilot_20260827.sh
