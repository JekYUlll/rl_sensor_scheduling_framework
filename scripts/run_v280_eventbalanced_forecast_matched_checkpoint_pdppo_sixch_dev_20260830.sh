#!/usr/bin/env bash
set -euo pipefail

# V280 is the protocol-corrected continuation of V279. Each policy is trained
# with the same-seed V279 truth/oracle/static-validation assets as its control
# source, so validation-only checkpoint selection has an actual scene to score.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$#" -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(6801 6802); fi

run_one() {
  local seed="$1"
  (
    export RUN_PREFIX="${RUN_PREFIX:-v280_eventbalanced_forecast_matched_checkpoint_pdppo_sixch_dev}"
    export LOG_PREFIX="${LOG_PREFIX:-v280_eventbalanced_forecast_matched_checkpoint_pdppo_sixch}"
    export REWARD_PROXY_MODE=forecast
    export EVENT_START_PROB=1.0
    export DECISION_ONLY_POLICY_UPDATES=0
    export DECISION_BLOCK_CREDIT=0
    export DECISION_BLOCK_REWARD_MODE=sum
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean
    export CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export CONTROL_SOURCE_RUN_DIR="reports/v279_eventbalanced_forecast_checkpoint_pdppo_sixch_dev_seed${seed}_b1p75_20260822"
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/${LOG_PREFIX:-v280_eventbalanced_forecast_matched_checkpoint_pdppo_sixch}_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for seed in "${SEEDS[@]}"; do run_one "$seed" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
