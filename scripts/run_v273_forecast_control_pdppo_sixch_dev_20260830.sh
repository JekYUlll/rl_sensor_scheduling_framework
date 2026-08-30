#!/usr/bin/env bash
set -euo pipefail

# Matched control for V272: retain its scene and PPO configuration, but use
# the original per-step forecast-loss reward and ordinary policy updates.
if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(6201 6202)
fi

REWARD_PROXY_MODE=forecast \
DECISION_ONLY_POLICY_UPDATES=0 \
DECISION_BLOCK_CREDIT=0 \
DECISION_BLOCK_REWARD_MODE=sum \
RUN_PREFIX="${RUN_PREFIX:-v273_forecast_control_pdppo_sixch_dev}" \
LOG_PREFIX="${LOG_PREFIX:-v273_forecast_control_pdppo_sixch}" \
bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "${SEEDS[@]}"
