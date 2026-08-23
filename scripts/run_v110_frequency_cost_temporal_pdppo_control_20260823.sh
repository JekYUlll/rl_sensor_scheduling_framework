#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Matched V104 seed-1304 control. The only method change is the GRU encoder over
# the existing online value and observation-mask history.
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1304}" \
GPU_IDS="${GPU_IDS:-0}" \
POLICY_SEED_OFFSET="${POLICY_SEED_OFFSET:-3000}" \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v110_frequency_cost_temporal_pdppo_control}" \
TEMPORAL_ENCODER_OVERRIDE=1 \
TEMPORAL_HIDDEN_DIM_OVERRIDE="${TEMPORAL_HIDDEN_DIM_OVERRIDE:-64}" \
bash scripts/run_v104_frequency_cost_pdppo_dev_20260823.sh
