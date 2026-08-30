#!/usr/bin/env bash
set -euo pipefail

# V285 tests a semi-Markov, action-aligned reward: at each genuine decision,
# reward the mean frozen-forecaster loss over the selected dwell block. It uses
# no bandit action, event label, or counterfactual label.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$#" -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(6901 6902); fi
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="v285_absolute_dwell_block_pdppo_sixch_dev"
    export LOG_PREFIX="v285_absolute_dwell_block_pdppo_sixch"
    export REWARD_PROXY_MODE=forecast_block_absolute
    export EVENT_START_PROB=1.0
    export DECISION_ONLY_POLICY_UPDATES=1
    export DECISION_BLOCK_CREDIT=0
    export DECISION_BLOCK_REWARD_MODE=sum
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean
    export CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export CONTROL_SOURCE_RUN_DIR="reports/v279_eventbalanced_forecast_checkpoint_pdppo_sixch_dev_seed${seed}_b1p75_20260822"
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v285_absolute_dwell_block_pdppo_sixch_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do
  run_one "${SEEDS[$i]}" "$((i + GPU_OFFSET))" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
