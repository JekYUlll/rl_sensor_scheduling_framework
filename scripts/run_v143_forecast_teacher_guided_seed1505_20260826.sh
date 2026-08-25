#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Training-only forecast teacher on policy-visited states. The executable policy
# remains a feasibility-masked PPO policy with no teacher input at evaluation.
SEEDS_OVERRIDE="1505" \
GPU_IDS="0" \
RUN_PREFIX_OVERRIDE=v143_forecast_teacher_guided_seed1505 \
BC_PRETRAIN_TARGET_MODE_OVERRIDE=hard \
FORECAST_VALUE_AUX_COEF_OVERRIDE=0 \
AWBC_COEF_OVERRIDE=0.5 \
AWBC_LABEL_STRIDE_OVERRIDE=4 \
ENT_COEF_OVERRIDE=0.005 \
bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh
