#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Clean policy-initialization control on the frozen V120/V132 development
# scenes. V132 supplies offset 12000; this runner adds offsets 13000 and 14000.
# Selection is performed only with the frozen calibration/validation score.
SEEDS="1301 1302 1303 1304 1305"
GPUS="0 1 2 3 4"

for offset in 13000 14000; do
  SEEDS_OVERRIDE="$SEEDS" \
  GPU_IDS="$GPUS" \
  POLICY_SEED_OFFSET="$offset" \
  RUN_PREFIX_OVERRIDE="v136_soft_onpolicy_temp075_init${offset}" \
  FORECAST_VALUE_AUX_LOSS_OVERRIDE=soft_ce \
  FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.75 \
  bash scripts/run_v125_dense_onpolicy_forecast_value_pilot_20260823.sh
done

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
"$PY" scripts/101_v32_select_policy_initialization.py \
  --run-glob 'reports/v132_soft_onpolicy_temp075_seed*_b1p75_20260822' \
  --run-glob 'reports/v136_soft_onpolicy_temp075_init13000_seed*_b1p75_20260822' \
  --run-glob 'reports/v136_soft_onpolicy_temp075_init14000_seed*_b1p75_20260822' \
  --out-root reports/aggregate/v136_soft_onpolicy_multiinit_dev_20260826
