#!/usr/bin/env bash
set -euo pipefail

# V342 is a bounded teacher-transfer diagnostic. It changes only the BC
# pretraining sample selection: forced dwell rows are excluded, while the
# forecast-value target, masked action space, scene, and evaluator match V341.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RUN_PREFIX_OVERRIDE="v342_decision_only_bc_diag"
export LOG_PREFIX_OVERRIDE="v342_decision_only_bc_diag"
export POLICY_SEEDS_OVERRIDE="7351 7352"
export BC_PRETRAIN_DECISION_ONLY_OVERRIDE=1
bash scripts/run_v341_recalibrated_scene_bc_only_pdppo_diag_20260901.sh
