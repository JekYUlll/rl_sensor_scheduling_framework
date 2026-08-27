#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1701}" \
GPU_IDS="${GPU_IDS:-0}" \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v187_v170_alertonly_mask_value_pilot}" \
LOG_DIR_OVERRIDE="${LOG_DIR_OVERRIDE:-logs/v187_v170_alertonly_mask_value_pilot}" \
FORECAST_VALUE_HEAD_IGNORE_QUALITY=1 \
bash scripts/run_v186_v170_ranked_mask_value_pilot_20260827.sh
