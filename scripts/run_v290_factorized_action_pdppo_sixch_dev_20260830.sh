#!/usr/bin/env bash
set -euo pipefail

# V290 replaces candidate-embedding action scoring with a structured policy:
# the actor emits one activation logit per channel, and each candidate subset
# receives the sum of its channel-wise Bernoulli log-probabilities before the
# exact feasible-action mask is applied. The forecast reward and scene are
# unchanged; no bandit or test information is used.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$#" -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(6801 6802); fi
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="v290_factorized_action_pdppo_sixch_dev"
    export LOG_PREFIX="v290_factorized_action_pdppo_sixch"
    export TOTAL_TIMESTEPS=30000
    export REWARD_PROXY_MODE=forecast_decision EVENT_START_PROB=1.0
    export DECISION_ONLY_POLICY_UPDATES=1 CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_RANKING_COEF=0
    export CONTROL_SOURCE_RUN_DIR="reports/v279_eventbalanced_forecast_checkpoint_pdppo_sixch_dev_seed${seed}_b1p75_20260822"
    export FACTORIZED_ACTION_POLICY=1
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v290_factorized_action_pdppo_sixch_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "${SEEDS[$i]}" "$((i + GPU_OFFSET))" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
