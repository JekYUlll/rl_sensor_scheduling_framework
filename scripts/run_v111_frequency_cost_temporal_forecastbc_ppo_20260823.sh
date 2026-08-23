#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Add forecast-loss PPO updates to the passing V109 representation and
# forecast-value BC initialization. Continuing imitation and subtype auxiliary
# losses are disabled to isolate sequential PPO transfer.
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1304}" \
GPU_IDS="${GPU_IDS:-0}" \
POLICY_SEED_OFFSET="${POLICY_SEED_OFFSET:-7000}" \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v111_frequency_cost_temporal_forecastbc_ppo}" \
TOTAL_TIMESTEPS_OVERRIDE="${TOTAL_TIMESTEPS_OVERRIDE:-40960}" \
BC_PRETRAIN_STEPS_OVERRIDE="${BC_PRETRAIN_STEPS_OVERRIDE:-4096}" \
BC_PRETRAIN_EPOCHS_OVERRIDE="${BC_PRETRAIN_EPOCHS_OVERRIDE:-20}" \
BC_PRETRAIN_TARGET_MODE_OVERRIDE=hard \
AWBC_TEACHER_MODE_OVERRIDE=oracle_greedy \
GREEDY_LOOKAHEAD_STEPS_OVERRIDE=8 \
AWBC_COEF_OVERRIDE=0 \
SUBTYPE_AUX_COEF_OVERRIDE=0 \
SUBTYPE_ACTION_CE_COEF_OVERRIDE=0 \
TEMPORAL_ENCODER_OVERRIDE=1 \
TEMPORAL_HIDDEN_DIM_OVERRIDE="${TEMPORAL_HIDDEN_DIM_OVERRIDE:-64}" \
bash scripts/run_v104_frequency_cost_pdppo_dev_20260823.sh
