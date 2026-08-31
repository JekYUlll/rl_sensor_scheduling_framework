#!/usr/bin/env bash
set -euo pipefail

# V322 combines the validated V319 hard forecast-value initialization with the
# existing temporal observation encoder.  It changes representation only:
# reward, feasible action masks, teacher target, dwell handling, and PPO budget
# remain those of V319.  No bandit-dependent signal or test-time label is used.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=(6811 6812)
POLICY_SEEDS=(${POLICY_SEEDS_OVERRIDE:-7021 7022})
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local idx="$1" seed="${SEEDS[$1]}" policy_seed="${POLICY_SEEDS[$1]}"
  local source="reports/v294_corrected_sixchannel_quality_pdppo_dev_seed${seed}_b1p75_20260822"
  (
    export CUDA_VISIBLE_DEVICES="$((idx + GPU_OFFSET))"
    export RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-v322_temporal_hardpretrain_pdppo_dev}"
    export LOG_PREFIX="${LOG_PREFIX_OVERRIDE:-v322_temporal_hardpretrain_pdppo}"
    export TOTAL_TIMESTEPS=50000 POLICY_SEED="$policy_seed" CONTROL_SOURCE_RUN_DIR="$source"
    export REWARD_PROXY_MODE=forecast REWARD_LOSS_NORMALIZATION=staticnorm_subtype EVENT_START_PROB=1.0
    export GREEDY_LOOKAHEAD_STEPS=6 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=6
    export DECISION_ONLY_POLICY_UPDATES=1 DECISION_BLOCK_CREDIT=0 DECISION_BLOCK_REWARD_MODE=sum
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=16384 BC_PRETRAIN_EPOCHS=20 BC_PRETRAIN_LOSS_COEF=1
    export BC_PRETRAIN_TARGET_MODE=hard_forecast_value BC_SOFT_TEMPERATURE=1.0
    export BC_PRETRAIN_DECISION_ONLY=0 BC_PRETRAIN_FORCED_ACTION_WEIGHT=1.0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_RANKING_COEF=0 CANDIDATE_INTERACTION_SCORE=0 FACTORIZED_ACTION_POLICY=0
    export AGENT_CONTEXT_COLUMNS="agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert" CONTEXT_FEATURE_DIM=20
    export TEMPORAL_ENCODER=1 TEMPORAL_HIDDEN_DIM=64 SEPARATE_ACTOR_CRITIC_GRAD_CLIP=1
    export ENT_COEF=0.02
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v322_temporal_hardpretrain_pdppo_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in 0 1; do run_one "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
