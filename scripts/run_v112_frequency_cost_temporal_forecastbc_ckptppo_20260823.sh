#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# V111 with calibration/validation checkpoint selection. The forecast-value BC
# checkpoint at PPO step zero and every fifth PPO update are eligible; final
# test remains untouched until the validation-selected checkpoint is frozen.
CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v112_frequency_cost_temporal_forecastbc_ckptppo}" \
bash scripts/run_v111_frequency_cost_temporal_forecastbc_ppo_20260823.sh
