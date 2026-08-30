#!/usr/bin/env bash
set -euo pipefail

# V278 tests a matched training-coverage hypothesis.  It restores the
# ordinary per-step forecast-loss reward and changes only the probability of
# selecting an event-containing training start from 0.70 to 1.00.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RUN_PREFIX="${RUN_PREFIX:-v278_eventbalanced_forecast_pdppo_sixch_dev}"
export LOG_PREFIX="${LOG_PREFIX:-v278_eventbalanced_forecast_pdppo_sixch}"
export REWARD_PROXY_MODE=forecast
export EVENT_START_PROB=1.0
export DECISION_ONLY_POLICY_UPDATES=0
export DECISION_BLOCK_CREDIT=0
export DECISION_BLOCK_REWARD_MODE=sum
exec bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$@"
