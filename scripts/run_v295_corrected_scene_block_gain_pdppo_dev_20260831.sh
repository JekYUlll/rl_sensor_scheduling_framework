#!/usr/bin/env bash
set -euo pipefail

# V295 tests a temporal block-gain reward on the corrected six-channel scene.
# All scene, action, cost, and evaluation settings remain those of V294; only
# the scalar prediction-loss reward is changed from point loss to block gain.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$#" -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(6813 6814); fi

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="v295_corrected_scene_block_gain_pdppo_dev"
    export LOG_PREFIX="v295_corrected_scene_block_gain_pdppo"
    export TOTAL_TIMESTEPS=50000
    export REWARD_PROXY_MODE=forecast_block_gain EVENT_START_PROB=1.0
    export DECISION_ONLY_POLICY_UPDATES=0 DECISION_BLOCK_CREDIT=0
    export DECISION_BLOCK_REWARD_MODE=sum
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_RANKING_COEF=0
    export CANDIDATE_INTERACTION_SCORE=0 FACTORIZED_ACTION_POLICY=0
    export AGENT_CONTEXT_COLUMNS="agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert"
    export CONTEXT_FEATURE_DIM=20
    export CONTROL_SOURCE_RUN_DIR=""
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v295_corrected_scene_block_gain_pdppo_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "${SEEDS[$i]}" "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
