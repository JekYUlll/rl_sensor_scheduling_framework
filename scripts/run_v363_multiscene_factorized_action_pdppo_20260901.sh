#!/usr/bin/env bash
set -euo pipefail

# V363 tests factorized channel logits on the frozen V357 cycling scenes.
# It retains V361/V362's multi-scene training, forecast reward, hard mask,
# checkpoint rule, and training budget. Only the actor action parameterization
# changes: channel-wise logits are composed over the feasible candidate masks.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v363_multiscene_factorized_action_pdppo}"
export LOG_PREFIX_OVERRIDE="${LOG_PREFIX_OVERRIDE:-v363_multiscene_factorized_action_pdppo}"
export POLICY_SEEDS_OVERRIDE="${POLICY_SEEDS_OVERRIDE:-7511 7512 7513 7514}"
export DIRECT_MASK_ACTION_PRIMARY_OVERRIDE=0
export FACTORIZED_ACTION_POLICY_OVERRIDE=1

exec bash scripts/run_v361_multiscene_cycling_pdppo_20260901.sh
