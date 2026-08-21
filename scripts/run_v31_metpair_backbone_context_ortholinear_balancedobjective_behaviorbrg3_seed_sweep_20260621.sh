#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg3}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolbrg3}"
export DATE_TAG="${DATE_TAG:-20260621}"

# BRG-3 keeps the successful BRG-2 deployable regime-belief/router path and
# adds a moderate action-fidelity signal. The target is seed92's learned-policy
# step gap: explicit subtype replay clears the seed, but PPO does not fully
# realize that dynamic schedule.
export INCLUDE_OBSERVABLE_REGIME_BELIEF="${INCLUDE_OBSERVABLE_REGIME_BELIEF:-1}"
export REGIME_BELIEF_LOOKBACK="${REGIME_BELIEF_LOOKBACK:-12}"
export EVENT_GATED_ACTOR="${EVENT_GATED_ACTOR:-1}"
export SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-5.0}"
export SUBTYPE_AUX_LOOKAHEAD_STEPS="${SUBTYPE_AUX_LOOKAHEAD_STEPS:-12}"
export SUBTYPE_ROUTER_MIN_CONFIDENCE="${SUBTYPE_ROUTER_MIN_CONFIDENCE:-0.70}"
export EVAL_SUBTYPE_ROUTER_MIN_CONFIDENCE="${EVAL_SUBTYPE_ROUTER_MIN_CONFIDENCE:-0.70}"
export ENT_COEF="${ENT_COEF:-0.0075}"

# Moderate, not BD-1-strength, because BD-1 failed without BRG's state path.
export SUBTYPE_ACTION_CE_COEF="${SUBTYPE_ACTION_CE_COEF:-0.25}"
export SUBTYPE_ACTION_MARGIN_COEF="${SUBTYPE_ACTION_MARGIN_COEF:-0.05}"
export SUBTYPE_ACTION_MARGIN="${SUBTYPE_ACTION_MARGIN:-0.50}"

export AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_auto}"
export AWBC_TEACHER_AUTO_SCORE_MODE="${AWBC_TEACHER_AUTO_SCORE_MODE:-raw}"

export LAMBDA_DUTY_BALANCE="${LAMBDA_DUTY_BALANCE:-0.0}"
export DUTY_SCORE_FEEDBACK="${DUTY_SCORE_FEEDBACK:-0.0}"
export DUTY_HARD_GUARD="${DUTY_HARD_GUARD:-0}"

bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_seed_sweep_20260620.sh "$@"
