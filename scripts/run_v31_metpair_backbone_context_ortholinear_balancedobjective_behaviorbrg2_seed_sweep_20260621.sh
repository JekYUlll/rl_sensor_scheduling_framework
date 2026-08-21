#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg2}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolbrg2}"
export DATE_TAG="${DATE_TAG:-20260621}"

# BRG-2 is a bounded follow-up to BRG-1, not a new broad sweep. It keeps PPO,
# the BO-1 met+specialist scene, and the current simulated sensor baseline. The
# intervention tests whether matching raw/eval subtype-router confidence closes
# the seed87 raw-deployment collapse observed in BRG-1.
export INCLUDE_OBSERVABLE_REGIME_BELIEF="${INCLUDE_OBSERVABLE_REGIME_BELIEF:-1}"
export REGIME_BELIEF_LOOKBACK="${REGIME_BELIEF_LOOKBACK:-12}"
export EVENT_GATED_ACTOR="${EVENT_GATED_ACTOR:-1}"
export SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-5.0}"
export SUBTYPE_AUX_LOOKAHEAD_STEPS="${SUBTYPE_AUX_LOOKAHEAD_STEPS:-12}"
export SUBTYPE_ROUTER_MIN_CONFIDENCE="${SUBTYPE_ROUTER_MIN_CONFIDENCE:-0.70}"
export EVAL_SUBTYPE_ROUTER_MIN_CONFIDENCE="${EVAL_SUBTYPE_ROUTER_MIN_CONFIDENCE:-0.70}"
export ENT_COEF="${ENT_COEF:-0.0075}"

# Keep BD-1's direct action CE/margin losses disabled. BRG-2 isolates the
# deployable regime-belief/router hypothesis after BD-1 produced a real negative
# result.
export SUBTYPE_ACTION_CE_COEF="${SUBTYPE_ACTION_CE_COEF:-0.0}"
export SUBTYPE_ACTION_MARGIN_COEF="${SUBTYPE_ACTION_MARGIN_COEF:-0.0}"
export SUBTYPE_ACTION_MARGIN="${SUBTYPE_ACTION_MARGIN:-0.70}"

export AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_auto}"
export AWBC_TEACHER_AUTO_SCORE_MODE="${AWBC_TEACHER_AUTO_SCORE_MODE:-raw}"

export LAMBDA_DUTY_BALANCE="${LAMBDA_DUTY_BALANCE:-0.0}"
export DUTY_SCORE_FEEDBACK="${DUTY_SCORE_FEEDBACK:-0.0}"
export DUTY_HARD_GUARD="${DUTY_HARD_GUARD:-0}"

bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_seed_sweep_20260620.sh "$@"
