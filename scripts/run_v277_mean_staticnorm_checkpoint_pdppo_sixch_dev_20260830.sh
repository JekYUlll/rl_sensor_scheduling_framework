#!/usr/bin/env bash
set -euo pipefail

# V277 uses the mean of ordinary and static-normalized macro ratios for
# validation checkpoint selection.  It is a bounded objective-balance test;
# training and online execution remain unchanged from V275/V276.
CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 \
CHECKPOINT_SELECTION_SCORE=mean_static_ratio \
CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1 \
REWARD_PROXY_MODE=forecast \
DECISION_ONLY_POLICY_UPDATES=0 \
DECISION_BLOCK_CREDIT=0 \
DECISION_BLOCK_REWARD_MODE=sum \
RUN_PREFIX="${RUN_PREFIX:-v277_mean_staticnorm_checkpoint_pdppo_sixch_dev}" \
LOG_PREFIX="${LOG_PREFIX:-v277_mean_staticnorm_checkpoint_pdppo_sixch}" \
bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$@"
