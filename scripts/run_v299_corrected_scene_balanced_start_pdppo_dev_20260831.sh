#!/usr/bin/env bash
set -euo pipefail

# V299 tests a less event-centered PPO training start distribution. It keeps
# V298's corrected scene, ordinary forecast reward, and active validation
# checkpoint selector; only event_start_prob changes from 1.0 to 0.35.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=(6811 6812)
POLICY_SEEDS=(6851 6852)
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local idx="$1" seed="${SEEDS[$1]}" policy_seed="${POLICY_SEEDS[$1]}"
  local source="reports/v294_corrected_sixchannel_quality_pdppo_dev_seed${seed}_b1p75_20260822"
  (
    export CUDA_VISIBLE_DEVICES="$((idx + GPU_OFFSET))"
    export RUN_PREFIX="v299_corrected_scene_balanced_start_pdppo_dev"
    export LOG_PREFIX="v299_corrected_scene_balanced_start_pdppo"
    export TOTAL_TIMESTEPS=50000
    export POLICY_SEED="$policy_seed"
    export CONTROL_SOURCE_RUN_DIR="$source"
    export REWARD_PROXY_MODE=forecast EVENT_START_PROB=0.35
    export DECISION_ONLY_POLICY_UPDATES=0 DECISION_BLOCK_CREDIT=0
    export DECISION_BLOCK_REWARD_MODE=sum
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_RANKING_COEF=0
    export CANDIDATE_INTERACTION_SCORE=0 FACTORIZED_ACTION_POLICY=0
    export AGENT_CONTEXT_COLUMNS="agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert"
    export CONTEXT_FEATURE_DIM=20
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v299_corrected_scene_balanced_start_pdppo_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in 0 1; do run_one "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
