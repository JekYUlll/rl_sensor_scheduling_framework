#!/usr/bin/env bash
set -euo pipefail

# V276 changes only validation checkpoint selection relative to V275.  The
# selected checkpoint minimizes the worse of ordinary and static-normalized
# macro ratios on same-seed validation assets.  Training still uses ordinary
# forecast reward and receives no comparator-dependent or test information.
CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 \
CHECKPOINT_SELECTION_SCORE=max_static_ratio \
CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1 \
REWARD_PROXY_MODE=forecast \
DECISION_ONLY_POLICY_UPDATES=0 \
DECISION_BLOCK_CREDIT=0 \
DECISION_BLOCK_REWARD_MODE=sum \
RUN_PREFIX="${RUN_PREFIX:-v276_staticnorm_checkpoint_pdppo_sixch_dev}" \
LOG_PREFIX="${LOG_PREFIX:-v276_staticnorm_checkpoint_pdppo_sixch}" \
bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$@"
