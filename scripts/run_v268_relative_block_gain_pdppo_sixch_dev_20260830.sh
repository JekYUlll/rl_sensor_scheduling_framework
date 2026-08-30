#!/usr/bin/env bash
set -euo pipefail

# V268 keeps V267's closed-loop block target but normalizes the action gain to
# a bounded relative improvement. It remains independent of bandit actions,
# teacher actions, and test labels.
if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(5801 5802)
fi

REWARD_PROXY_MODE=forecast_block_relative_gain \
RUN_PREFIX="${RUN_PREFIX:-v268_relative_block_gain_pdppo_sixch_dev}" \
LOG_PREFIX="${LOG_PREFIX:-v268_relative_block_gain_pdppo_sixch}" \
bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "${SEEDS[@]}"
