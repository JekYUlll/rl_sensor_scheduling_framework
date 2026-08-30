#!/usr/bin/env bash
set -euo pipefail

# V288 tests forecast-value regression as a training-partition policy
# initialization. PPO, forecast reward, and feasibility masking remain the
# same as V287; no bandit action or test information is used.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$#" -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(6801 6802); fi
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="v288_forecast_value_pretrain_pdppo_sixch_dev"
    export LOG_PREFIX="v288_forecast_value_pretrain_pdppo_sixch"
    export TOTAL_TIMESTEPS=30000
    export REWARD_PROXY_MODE=forecast_decision EVENT_START_PROB=1.0
    export DECISION_ONLY_POLICY_UPDATES=1 CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=4096 BC_PRETRAIN_EPOCHS=20 BC_PRETRAIN_LOSS_COEF=1.0
    export BC_PRETRAIN_TARGET_MODE=forecast_value_regression BC_SOFT_TEMPERATURE=0.75
    export GREEDY_LOOKAHEAD_STEPS=6
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_AUX_STRIDE=32
    export FORECAST_VALUE_AUX_LOOKAHEAD_STEPS=6 FORECAST_VALUE_AUX_LOSS=soft_ce
    export FORECAST_VALUE_AUX_TEMPERATURE=1.0 FORECAST_VALUE_RANKING_COEF=0.0
    export CONTROL_SOURCE_RUN_DIR="reports/v279_eventbalanced_forecast_checkpoint_pdppo_sixch_dev_seed${seed}_b1p75_20260822"
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v288_forecast_value_pretrain_pdppo_sixch_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "${SEEDS[$i]}" "$((i + GPU_OFFSET))" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
