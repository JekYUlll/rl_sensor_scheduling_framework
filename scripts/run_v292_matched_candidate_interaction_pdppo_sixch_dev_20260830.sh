#!/usr/bin/env bash
set -euo pipefail

# V292 isolates candidate interaction against the V279 configuration. It
# changes only the candidate-interaction actor term; reward mode, update
# convention, training length, scene, evaluator, and information boundary
# remain matched to V279.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$#" -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(6801 6802); fi
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="v292_matched_candidate_interaction_pdppo_sixch_dev"
    export LOG_PREFIX="v292_matched_candidate_interaction_pdppo_sixch"
    export TOTAL_TIMESTEPS=50000
    export REWARD_PROXY_MODE=forecast EVENT_START_PROB=1.0
    export DECISION_ONLY_POLICY_UPDATES=0 CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_RANKING_COEF=0
    export CANDIDATE_INTERACTION_SCORE=1 FACTORIZED_ACTION_POLICY=0
    export CONTROL_SOURCE_RUN_DIR="reports/v279_eventbalanced_forecast_checkpoint_pdppo_sixch_dev_seed${seed}_b1p75_20260822"
    bash scripts/run_v267_block_gain_pdppo_sixch_dev.sh "$seed"
  ) >"logs/${LOG_PREFIX}_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "${SEEDS[$i]}" "$((i + GPU_OFFSET))" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
