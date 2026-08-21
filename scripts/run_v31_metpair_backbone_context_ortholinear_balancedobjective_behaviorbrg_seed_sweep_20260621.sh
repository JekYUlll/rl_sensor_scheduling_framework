#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolbrg}"
export DATE_TAG="${DATE_TAG:-20260621}"

# BRG-1 keeps PPO, the BO-1 met+specialist scene, and the existing simulated
# sensor roles. The intervention is a deployable state/architecture path:
# expose observable regime-belief features to PPO, strengthen subtype belief
# learning, and keep a conservative subtype-router deployment head.
export INCLUDE_OBSERVABLE_REGIME_BELIEF="${INCLUDE_OBSERVABLE_REGIME_BELIEF:-1}"
export REGIME_BELIEF_LOOKBACK="${REGIME_BELIEF_LOOKBACK:-12}"
export EVENT_GATED_ACTOR="${EVENT_GATED_ACTOR:-1}"
export SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-5.0}"
export SUBTYPE_AUX_LOOKAHEAD_STEPS="${SUBTYPE_AUX_LOOKAHEAD_STEPS:-12}"
export SUBTYPE_ROUTER_MIN_CONFIDENCE="${SUBTYPE_ROUTER_MIN_CONFIDENCE:-0.45}"
export EVAL_SUBTYPE_ROUTER_MIN_CONFIDENCE="${EVAL_SUBTYPE_ROUTER_MIN_CONFIDENCE:-0.70}"
export ENT_COEF="${ENT_COEF:-0.005}"

# Do not inherit BD-1's direct action CE/margin shaping by default. BRG-1 tests
# whether a better regime representation fixes deployment behavior.
export SUBTYPE_ACTION_CE_COEF="${SUBTYPE_ACTION_CE_COEF:-0.0}"
export SUBTYPE_ACTION_MARGIN_COEF="${SUBTYPE_ACTION_MARGIN_COEF:-0.0}"
export SUBTYPE_ACTION_MARGIN="${SUBTYPE_ACTION_MARGIN:-0.70}"

export AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_auto}"
export AWBC_TEACHER_AUTO_SCORE_MODE="${AWBC_TEACHER_AUTO_SCORE_MODE:-raw}"

# Keep duty feedback off; if BRG-1 fails, duty constraints remain a separate
# closed direction rather than a hidden source of apparent behavior.
export LAMBDA_DUTY_BALANCE="${LAMBDA_DUTY_BALANCE:-0.0}"
export DUTY_SCORE_FEEDBACK="${DUTY_SCORE_FEEDBACK:-0.0}"
export DUTY_HARD_GUARD="${DUTY_HARD_GUARD:-0}"

bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_seed_sweep_20260620.sh "$@"
