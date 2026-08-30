#!/usr/bin/env bash
set -euo pipefail

# V245 tests the existing candidate-conditioned on-policy value head.  It is
# trained from PPO returns and does not use bandit actions, counterfactual
# labels, or privileged final-test information.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RUN_PREFIX="v245_onpolicy_action_value_pdppo_dev"
export LOG_PREFIX="v245_onpolicy_action_value_pdppo"
export ONPOLICY_ACTION_VALUE_COEF=0.10
export ONPOLICY_ACTION_VALUE_SCALE=0.50
exec bash scripts/run_v243_full_pdppo_no_awbc_balanced_quality_20260830.sh "${@:-3601 3602 3603 3604 3605}"
