#!/usr/bin/env bash
set -euo pipefail

# V279 is the protocol-corrected V278 control. It changes no training or
# scene parameter beyond event-start coverage, and restores validation-only
# checkpoint selection so the comparison is interpretable.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RUN_PREFIX="${RUN_PREFIX:-v279_eventbalanced_forecast_checkpoint_pdppo_sixch_dev}"
export LOG_PREFIX="${LOG_PREFIX:-v279_eventbalanced_forecast_checkpoint_pdppo_sixch}"
export REWARD_PROXY_MODE=forecast
export EVENT_START_PROB=1.0
export DECISION_ONLY_POLICY_UPDATES=0
export DECISION_BLOCK_CREDIT=0
export DECISION_BLOCK_REWARD_MODE=sum
export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean
export CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
exec bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$@"
