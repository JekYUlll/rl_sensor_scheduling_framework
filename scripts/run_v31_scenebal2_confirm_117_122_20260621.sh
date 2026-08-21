#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260621}"
SOURCE_DATE_TAG="${SOURCE_DATE_TAG:-${DATE_TAG}}"
TRAIN_SEEDS_TEXT="${TRAIN_SEEDS_TEXT:-118 119 120 121}"
ALL_SEEDS_TEXT="${ALL_SEEDS_TEXT:-117 118 119 120 121 122}"
GPU_IDS_TEXT="${SCENEBAL2_GPU_IDS:-0 1 2 3}"

RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2}"
BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolscbal2}"
ROUTER_CONF="${ROUTER_CONF:-0.5}"
EVAL_DIR="${EVAL_DIR:-eval_router_conf05_scenebal2_20260621}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-behavior_audit_router_conf05_scenebal2_20260621}"

# shellcheck disable=SC2206
TRAIN_SEEDS=(${TRAIN_SEEDS_TEXT})
# shellcheck disable=SC2206
ALL_SEEDS=(${ALL_SEEDS_TEXT})

if [[ "${#TRAIN_SEEDS[@]}" -eq 0 ]]; then
  echo "No train seeds configured" >&2
  exit 2
fi
if [[ "${#ALL_SEEDS[@]}" -eq 0 ]]; then
  echo "No aggregate seeds configured" >&2
  exit 2
fi

TRAIN_LABEL="$(IFS=_; echo "${TRAIN_SEEDS[*]}")"
ALL_LABEL="$(IFS=_; echo "${ALL_SEEDS[*]}")"

echo "scenebal2_confirm_start date=$(date -Is) train_seeds=${TRAIN_SEEDS[*]} all_seeds=${ALL_SEEDS[*]} router_conf=${ROUTER_CONF}"

SCENEBAL2_GPU_IDS="${GPU_IDS_TEXT}" \
RUN_PREFIX="${RUN_PREFIX}" \
BUDGET_LABEL="${BUDGET_LABEL}" \
ROUTER_CONF="${ROUTER_CONF}" \
EVAL_DIR="${EVAL_DIR}" \
BEHAVIOR_DIR="${BEHAVIOR_DIR}" \
AGG_LABEL="scenebal2_confirm_conf05_train_${TRAIN_LABEL}" \
DECISION_LABEL="SCENEBAL-2 Train-Seed Router-Conf0.5 ${TRAIN_LABEL}" \
DATE_TAG="${DATE_TAG}" \
PY="${PY}" \
bash scripts/run_v31_scenebal2_pivot_pilot_117_122_20260621.sh "${TRAIN_SEEDS[@]}"

SCENEBAL_ROUTER_GPU_IDS="${GPU_IDS_TEXT}" \
RUN_PREFIX="${RUN_PREFIX}" \
BUDGET_LABEL="${BUDGET_LABEL}" \
ROUTER_CONF="${ROUTER_CONF}" \
EVAL_DIR="${EVAL_DIR}" \
BEHAVIOR_DIR="${BEHAVIOR_DIR}" \
AGG_LABEL="scenebal2_confirm_conf05_${ALL_LABEL}" \
DECISION_LABEL="SCENEBAL-2 Fresh Confirmation Router-Conf0.5 ${ALL_LABEL}" \
DATE_TAG="${DATE_TAG}" \
SOURCE_DATE_TAG="${SOURCE_DATE_TAG}" \
PY="${PY}" \
bash scripts/run_v31_scenebal1_router_conf05_reaudit_20260621.sh "${ALL_SEEDS[@]}"

echo "scenebal2_confirm_done date=$(date -Is) aggregate_label=scenebal2_confirm_conf05_${ALL_LABEL}"
