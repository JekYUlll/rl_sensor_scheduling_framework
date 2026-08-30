#!/usr/bin/env bash
set -euo pipefail

# V274 tests standard validation-based checkpoint selection.  It keeps the
# ordinary forecast reward and V273 architecture, but selects a checkpoint
# using calibration/validation forecast score subject to the existing
# behavior-validity gate.  No comparator, test label, or bandit signal enters
# training or selection.
CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 \
CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1 \
REWARD_PROXY_MODE=forecast \
DECISION_ONLY_POLICY_UPDATES=0 \
DECISION_BLOCK_CREDIT=0 \
DECISION_BLOCK_REWARD_MODE=sum \
RUN_PREFIX="${RUN_PREFIX:-v274_validation_checkpoint_pdppo_sixch_dev}" \
LOG_PREFIX="${LOG_PREFIX:-v274_validation_checkpoint_pdppo_sixch}" \
bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$@"
