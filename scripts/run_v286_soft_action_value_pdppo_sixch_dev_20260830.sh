#!/usr/bin/env bash
set -euo pipefail

# V286 adds a forecast-derived soft action-ranking auxiliary to ordinary
# decision-time PPO. It does not use bandit actions, event labels, or test data.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$#" -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(7001 7002); fi
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="v286_soft_action_value_pdppo_sixch_dev_opt"
    export LOG_PREFIX="v286_soft_action_value_pdppo_sixch_opt"
    export REWARD_PROXY_MODE=forecast_decision
    export EVENT_START_PROB=1.0
    export DECISION_ONLY_POLICY_UPDATES=1
    export DECISION_BLOCK_CREDIT=0
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean
    export CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export FORECAST_VALUE_AUX_COEF=0.5
    export FORECAST_VALUE_AUX_STRIDE=1
    export FORECAST_VALUE_AUX_LOOKAHEAD_STEPS=6
    export FORECAST_VALUE_AUX_LOSS=soft_ce
    export FORECAST_VALUE_AUX_TEMPERATURE=0.75
    export FORECAST_VALUE_RANKING_COEF=0.0
    export CONTROL_SOURCE_RUN_DIR="reports/v279_eventbalanced_forecast_checkpoint_pdppo_sixch_dev_seed${seed}_b1p75_20260822"
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v286_soft_action_value_pdppo_sixch_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do
  run_one "${SEEDS[$i]}" "$((i + GPU_OFFSET))" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
