#!/usr/bin/env bash
set -euo pipefail

# V333 repeats V332 after aligning one-step forecast-gain credit with the
# environment's minimum-dwell execution semantics: forced hold steps receive
# no action-dependent gain.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=(6851 6852)
POLICY_SEEDS=(${POLICY_SEEDS_OVERRIDE:-7211 7212})
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local idx="$1" seed="${SEEDS[$1]}" policy_seed="${POLICY_SEEDS[$1]}"
  (
    export CUDA_VISIBLE_DEVICES="$((idx + GPU_OFFSET))"
    export RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-v333_forecast_gain_guard_pdppo_dev}"
    export LOG_PREFIX="${LOG_PREFIX_OVERRIDE:-v333_forecast_gain_guard_pdppo}"
    export TOTAL_TIMESTEPS=50000 POLICY_SEED="$policy_seed" CONTROL_SOURCE_RUN_DIR=""
    export REWARD_PROXY_MODE=forecast_gain REWARD_LOSS_NORMALIZATION=none EVENT_START_PROB=1.0
    export GREEDY_LOOKAHEAD_STEPS=6 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=6
    export DECISION_ONLY_POLICY_UPDATES=0 DECISION_BLOCK_CREDIT=0 DECISION_BLOCK_REWARD_MODE=sum
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 CHECKPOINT_SELECTION_MIN_UPDATE=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_AUX_STRIDE=4 FORECAST_VALUE_AUX_LOOKAHEAD_STEPS=6
    export FORECAST_VALUE_AUX_LOSS=smooth_l1 FORECAST_VALUE_RANKING_COEF=0
    export FORECAST_VALUE_HEAD=0 FORECAST_VALUE_HEAD_MODE=independent FORECAST_VALUE_HEAD_SCALE=1.0 FORECAST_VALUE_HEAD_HIDDEN_DIM=128
    export FORECAST_VALUE_TRUST_GATE=0 FORECAST_VALUE_TRUST_HIDDEN_DIM=64
    export CANDIDATE_INTERACTION_SCORE=0 FACTORIZED_ACTION_POLICY=0
    export QUALITY_CONTEXT_ACTION_SCORE=0 ALIGNED_QUALITY_ACTION_SCORE=0 QUALITY_CONTEXT_POOLING=mean
    export AGENT_CONTEXT_COLUMNS="agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert" CONTEXT_FEATURE_DIM=20
    export TEMPORAL_ENCODER=0 TEMPORAL_HIDDEN_DIM=64 SEPARATE_ACTOR_CRITIC_GRAD_CLIP=1 ENT_COEF=0.02
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v333_forecast_gain_guard_pdppo_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in 0 1; do run_one "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
