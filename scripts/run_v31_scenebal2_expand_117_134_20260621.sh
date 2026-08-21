#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export DATE_TAG="${DATE_TAG:-20260621}"
export TRAIN_SEEDS_TEXT="${TRAIN_SEEDS_TEXT:-129 130 131 132 133 134}"
export ALL_SEEDS_TEXT="${ALL_SEEDS_TEXT:-117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134}"
export SCENEBAL2_GPU_IDS="${SCENEBAL2_GPU_IDS:-0 1 2 3 4 5}"

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolscbal2}"
export ROUTER_CONF="${ROUTER_CONF:-0.5}"
export EVAL_DIR="${EVAL_DIR:-eval_router_conf05_scenebal2_20260621}"
export BEHAVIOR_DIR="${BEHAVIOR_DIR:-behavior_audit_router_conf05_scenebal2_20260621}"

bash scripts/run_v31_scenebal2_confirm_117_122_20260621.sh
