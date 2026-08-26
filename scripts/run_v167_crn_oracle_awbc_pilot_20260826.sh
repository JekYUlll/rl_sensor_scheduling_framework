#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

AWBC_COEF_OVERRIDE=0.1 \
AWBC_LABEL_STRIDE_OVERRIDE=4 \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1601}" \
GPU_IDS="${GPU_IDS:-0}" \
RUN_PREFIX_OVERRIDE=v167_crn_oracle_awbc_pilot \
LOG_DIR_OVERRIDE=logs/v167_crn_oracle_awbc_pilot \
bash scripts/run_v163_crn_quality_pdppo_20260826.sh
