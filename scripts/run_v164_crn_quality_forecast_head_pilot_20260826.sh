#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

FORECAST_VALUE_HEAD=1 \
FORECAST_VALUE_HEAD_MODE=factorized \
FORECAST_VALUE_HEAD_SCALE=1.0 \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1601}" \
GPU_IDS="${GPU_IDS:-0}" \
RUN_PREFIX_OVERRIDE=v164_crn_quality_factorized_forecast_head \
LOG_DIR_OVERRIDE=logs/v164_crn_quality_factorized_forecast_head \
bash scripts/run_v163_crn_quality_pdppo_20260826.sh
