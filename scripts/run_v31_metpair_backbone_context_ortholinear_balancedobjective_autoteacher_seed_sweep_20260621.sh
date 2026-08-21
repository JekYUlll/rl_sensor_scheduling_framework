#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_autoteacher}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolboauto}"
export DATE_TAG="${DATE_TAG:-20260621}"

# Keep the BO-1 scenario/objective fixed, but replace the hand-written
# subtype AWBC teacher with a data-driven teacher selected from static-candidate
# calm/subtype losses on the validation/static-selection split.
export AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_static_auto}"
export AWBC_TEACHER_AUTO_SCORE_MODE="${AWBC_TEACHER_AUTO_SCORE_MODE:-raw}"

bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_seed_sweep_20260620.sh "$@"
