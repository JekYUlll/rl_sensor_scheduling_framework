#!/usr/bin/env bash
set -euo pipefail

# V337 quantifies policy-seed variance in the clean V319 configuration.  It
# repeats each corrected scene twice with fresh policy seeds, keeping the
# scene, reward, action geometry, and checkpoint protocol fixed.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=(6811 6812)
POLICY_SEEDS=(7261 7262 7263 7264)
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local slot="$1" seed="$2" policy_seed="$3"
  (
    export CUDA_VISIBLE_DEVICES="$((slot + GPU_OFFSET))"
    export RUN_PREFIX="v337_policy${policy_seed}_scene${seed}_pdppo_dev"
    export LOG_PREFIX="v337_policy${policy_seed}_scene${seed}_pdppo"
    export TOTAL_TIMESTEPS=50000 POLICY_SEED="$policy_seed"
    export CONTROL_SOURCE_RUN_DIR="reports/v294_corrected_sixchannel_quality_pdppo_dev_seed${seed}_b1p75_20260822"
    export REWARD_PROXY_MODE=forecast REWARD_LOSS_NORMALIZATION=staticnorm_subtype EVENT_START_PROB=1.0
    export GREEDY_LOOKAHEAD_STEPS=6 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=6
    export DECISION_ONLY_POLICY_UPDATES=1 DECISION_BLOCK_CREDIT=0 DECISION_BLOCK_REWARD_MODE=sum
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 CHECKPOINT_SELECTION_MIN_UPDATE=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=16384 BC_PRETRAIN_EPOCHS=20 BC_PRETRAIN_LOSS_COEF=1
    export BC_PRETRAIN_TARGET_MODE=hard_forecast_value BC_SOFT_TEMPERATURE=1.0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_RANKING_COEF=0 CANDIDATE_INTERACTION_SCORE=0 FACTORIZED_ACTION_POLICY=0
    export FORECAST_VALUE_HEAD=0 FORECAST_VALUE_HEAD_MODE=independent FORECAST_VALUE_HEAD_SCALE=0.0
    export AGENT_CONTEXT_COLUMNS="agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert" CONTEXT_FEATURE_DIM=20
    export TEMPORAL_ENCODER=1 TEMPORAL_HIDDEN_DIM=64 SEPARATE_ACTOR_CRITIC_GRAD_CLIP=1 ENT_COEF=0.02
    export QUALITY_CONTEXT_ACTION_SCORE=0 ALIGNED_QUALITY_ACTION_SCORE=0 QUALITY_CONTEXT_POOLING=mean
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v337_policy${policy_seed}_scene${seed}.log" 2>&1
}

run_batch() {
  local first="$1" second="$2"
  run_one 0 "${SEEDS[$((first % 2))]}" "${POLICY_SEEDS[$first]}" & local p1=$!
  run_one 1 "${SEEDS[$((second % 2))]}" "${POLICY_SEEDS[$second]}" & local p2=$!
  wait "$p1" "$p2"
}

mkdir -p logs
run_batch 0 1
run_batch 2 3
