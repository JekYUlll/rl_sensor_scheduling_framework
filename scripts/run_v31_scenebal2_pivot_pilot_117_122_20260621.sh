#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260621}"
SEEDS=("$@")
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  SEEDS=(122 117)
fi
if [[ -z "${SEED_LABEL:-}" ]]; then
  SEED_LABEL="$(IFS=_; echo "${SEEDS[*]}")"
fi

GPU_IDS_TEXT="${SCENEBAL2_GPU_IDS:-0 1}"
RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2}"
BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolscbal2}"
ROUTER_CONF="${ROUTER_CONF:-0.5}"
EVAL_DIR="${EVAL_DIR:-eval_router_conf05_scenebal2_20260621}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-behavior_audit_router_conf05_scenebal2_20260621}"
AGG_LABEL="${AGG_LABEL:-scenebal2_pivot_conf05_${SEED_LABEL}}"

mkdir -p logs

echo "scenebal2_pivot_start date=$(date -Is) seeds=${SEEDS[*]} router_conf=${ROUTER_CONF}"

# shellcheck disable=SC2206
GPU_IDS=(${GPU_IDS_TEXT})
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPU ids configured" >&2
  exit 2
fi

pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu_id="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  log_file="logs/scenebal2_pivot_seed${seed}_${DATE_TAG}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export SKIP_COLLECT=1
    export DEVICE="${DEVICE:-cuda}"
    export ORACLE_DEVICE="${ORACLE_DEVICE:-auto}"
    export ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
    export EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-cpu}"
    echo "scenebal2_seed_start seed=${seed} gpu=${gpu_id} date=$(date -Is)"
    bash scripts/run_v31_scenebal2_seed_sweep_20260621.sh "${seed}"
    echo "scenebal2_seed_done seed=${seed} gpu=${gpu_id} date=$(date -Is)"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done
if [[ "${rc}" -ne 0 ]]; then
  echo "scenebal2_pivot_training_failed date=$(date -Is)" >&2
  exit "${rc}"
fi

SCENEBAL_ROUTER_GPU_IDS="${GPU_IDS_TEXT}" \
RUN_PREFIX="${RUN_PREFIX}" \
BUDGET_LABEL="${BUDGET_LABEL}" \
ROUTER_CONF="${ROUTER_CONF}" \
EVAL_DIR="${EVAL_DIR}" \
BEHAVIOR_DIR="${BEHAVIOR_DIR}" \
AGG_LABEL="${AGG_LABEL}" \
DECISION_LABEL="SCENEBAL-2 Pivot Router-Conf0.5 Pilot ${SEED_LABEL}" \
DATE_TAG="${DATE_TAG}" \
PY="${PY}" \
bash scripts/run_v31_scenebal1_router_conf05_reaudit_20260621.sh "${SEEDS[@]}"

echo "scenebal2_pivot_done date=$(date -Is) aggregate_label=${AGG_LABEL}"
