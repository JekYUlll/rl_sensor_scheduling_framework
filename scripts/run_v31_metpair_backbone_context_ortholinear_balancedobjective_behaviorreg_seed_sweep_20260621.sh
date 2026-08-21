#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorreg}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolboreg}"
export DATE_TAG="${DATE_TAG:-20260621}"

# AT-2 keeps the BO-1 scene, sensor setup, and PPO scheduler, but moves the
# intervention from teacher selection to behavior robustness. The regularizer is
# intentionally mild: it discourages persistent specialist collapse without
# enforcing a deterministic round-robin schedule.
export AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_auto}"
export AWBC_TEACHER_AUTO_SCORE_MODE="${AWBC_TEACHER_AUTO_SCORE_MODE:-raw}"
export LAMBDA_DUTY_BALANCE="${LAMBDA_DUTY_BALANCE:-0.01}"
export DUTY_BALANCE_LOW="${DUTY_BALANCE_LOW:-0.04}"
export DUTY_BALANCE_HIGH="${DUTY_BALANCE_HIGH:-0.90}"
export DUTY_BALANCE_GRACE_STEPS="${DUTY_BALANCE_GRACE_STEPS:-192}"
export DUTY_SCORE_FEEDBACK="${DUTY_SCORE_FEEDBACK:-0.20}"
export DUTY_SCORE_TARGET="${DUTY_SCORE_TARGET:-0.32}"
export DUTY_HARD_GUARD="${DUTY_HARD_GUARD:-0}"
export DUTY_HARD_LOW="${DUTY_HARD_LOW:-0.04}"
export DUTY_HARD_HIGH="${DUTY_HARD_HIGH:-0.90}"
export DUTY_HARD_SCORE="${DUTY_HARD_SCORE:-2.5}"
export ENT_COEF="${ENT_COEF:-0.004}"

bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_seed_sweep_20260620.sh "$@"
