#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbd}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolbd}"
export DATE_TAG="${DATE_TAG:-20260621}"

# BD-1 keeps PPO, the BO-1 met+specialist scene, and the existing sensor setup.
# The intervention is an explicit state-dependent behaviour signal: subtype
# labels supervise the action distribution directly, in addition to the existing
# advantage-weighted teacher and subtype auxiliary head. This targets collapse
# where the policy predicts the subtype but still deploys a weakly differentiated
# mask.
export AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_auto}"
export AWBC_TEACHER_AUTO_SCORE_MODE="${AWBC_TEACHER_AUTO_SCORE_MODE:-raw}"
export SUBTYPE_ACTION_CE_COEF="${SUBTYPE_ACTION_CE_COEF:-0.50}"
export SUBTYPE_ACTION_MARGIN_COEF="${SUBTYPE_ACTION_MARGIN_COEF:-0.10}"
export SUBTYPE_ACTION_MARGIN="${SUBTYPE_ACTION_MARGIN:-0.70}"
export ENT_COEF="${ENT_COEF:-0.005}"

# Do not carry over BR-1 duty feedback by default; BD-1 should isolate whether
# action-level subtype separation fixes behavior without a hidden rotation guard.
export LAMBDA_DUTY_BALANCE="${LAMBDA_DUTY_BALANCE:-0.0}"
export DUTY_SCORE_FEEDBACK="${DUTY_SCORE_FEEDBACK:-0.0}"
export DUTY_HARD_GUARD="${DUTY_HARD_GUARD:-0}"

bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_seed_sweep_20260620.sh "$@"
