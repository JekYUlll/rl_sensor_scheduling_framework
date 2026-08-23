#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Matched V107 hard-label observability probe with a GRU encoder over the
# existing online history and observation-mask sequence.
POLICY_SEED_OFFSET="${POLICY_SEED_OFFSET:-7000}" \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v109_frequency_cost_temporal_observability_probe}" \
TEMPORAL_ENCODER_OVERRIDE=1 \
TEMPORAL_HIDDEN_DIM_OVERRIDE="${TEMPORAL_HIDDEN_DIM_OVERRIDE:-64}" \
bash scripts/run_v107_frequency_cost_observability_probe_20260823.sh
