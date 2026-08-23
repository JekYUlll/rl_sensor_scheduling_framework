#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# A training-partition-only observability diagnostic. The policy receives the
# normal online state while privileged eight-step forecast values provide hard
# action labels during BC pretraining. PPO updates and auxiliary labels are off.
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1304}" \
GPU_IDS="${GPU_IDS:-0}" \
POLICY_SEED_OFFSET="${POLICY_SEED_OFFSET:-5000}" \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v107_frequency_cost_observability_probe}" \
TOTAL_TIMESTEPS_OVERRIDE=0 \
BC_PRETRAIN_STEPS_OVERRIDE="${BC_PRETRAIN_STEPS_OVERRIDE:-4096}" \
BC_PRETRAIN_EPOCHS_OVERRIDE="${BC_PRETRAIN_EPOCHS_OVERRIDE:-20}" \
BC_PRETRAIN_TARGET_MODE_OVERRIDE=hard \
AWBC_TEACHER_MODE_OVERRIDE=oracle_greedy \
GREEDY_LOOKAHEAD_STEPS_OVERRIDE=8 \
AWBC_COEF_OVERRIDE=0 \
SUBTYPE_AUX_COEF_OVERRIDE=0 \
SUBTYPE_ACTION_CE_COEF_OVERRIDE=0 \
bash scripts/run_v104_frequency_cost_pdppo_dev_20260823.sh
