#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260621}"
RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal1}"
BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolscbal1}"
SOURCE_DATE_TAG="${SOURCE_DATE_TAG:-20260621}"
ROUTER_CONF="${ROUTER_CONF:-0.5}"
EVAL_DIR="${EVAL_DIR:-eval_router_conf05_20260621}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-behavior_audit_router_conf05_20260621}"
AGG_LABEL="${AGG_LABEL:-scenebal1_24seed_93_116_router_conf05}"
DECISION_LABEL="${DECISION_LABEL:-Final Specialist-Budget Router-Conf0.5 Reaudit}"
GPU_IDS_TEXT="${SCENEBAL_ROUTER_GPU_IDS:-${SCENEBAL1_GPU_IDS:-0 1 2 3 4 5}}"
ORACLE_DEVICE="${ORACLE_DEVICE:-cpu}"
DEVICE="${DEVICE:-cuda}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116)
fi

if [[ -z "${SEED_LABEL:-}" ]]; then
  SEED_LABEL="$(IFS=_; echo "${SEEDS[*]}")"
fi

mkdir -p logs

echo "scenebal1_router_reaudit_start date=$(date -Is) seeds=${SEEDS[*]} router_conf=${ROUTER_CONF} eval_dir=${EVAL_DIR}"

# Use one worker per GPU. Each worker processes its round-robin slice
# sequentially, avoiding four concurrent evals on the same card.
# shellcheck disable=SC2206
GPU_IDS=(${GPU_IDS_TEXT})
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPU ids configured" >&2
  exit 2
fi

pids=()
for worker_idx in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$worker_idx]}"
  log_file="logs/scenebal1_router_conf05_worker${worker_idx}_${DATE_TAG}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    echo "worker_start idx=${worker_idx} gpu=${gpu_id} date=$(date -Is)"
    for seed_idx in "${!SEEDS[@]}"; do
      if (( seed_idx % ${#GPU_IDS[@]} != worker_idx )); then
        continue
      fi
      seed="${SEEDS[$seed_idx]}"
      run_dir="reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${SOURCE_DATE_TAG}"
      if [[ ! -d "${run_dir}" ]]; then
        echo "missing_run_dir seed=${seed} run_dir=${run_dir}" >&2
        exit 3
      fi
      echo "seed_start seed=${seed} gpu=${gpu_id} run_dir=${run_dir} date=$(date -Is)"
      if [[ ! -f "${run_dir}/${EVAL_DIR}/v2_custom_ppo_metrics.csv" ]]; then
        "${PY}" scripts/64_v31_eval_saved_run_operational_baselines.py \
          --source-run-dir "${run_dir}" \
          --out-dir "${run_dir}/${EVAL_DIR}" \
          --device "${DEVICE}" \
          --oracle-device "${ORACLE_DEVICE}" \
          --subtype-router \
          --subtype-router-min-confidence "${ROUTER_CONF}" \
          --skip-rollout-evaluation \
          2>&1 | tee "${run_dir}/${EVAL_DIR}.log"
      else
        echo "eval_exists seed=${seed} path=${run_dir}/${EVAL_DIR}/v2_custom_ppo_metrics.csv"
      fi
      if [[ ! -f "${run_dir}/${BEHAVIOR_DIR}/behavior_complexity_summary.json" ]]; then
        "${PY}" scripts/71_v31_behavior_complexity_audit.py \
          --out-dir "${run_dir}/${BEHAVIOR_DIR}" \
          "${run_dir}/${EVAL_DIR}/rollout_custom_ppo.npz" \
          "${run_dir}/${EVAL_DIR}/rollout_validation_selected_static.npz" \
          2>&1 | tee "${run_dir}/${BEHAVIOR_DIR}.log"
      else
        echo "behavior_exists seed=${seed} path=${run_dir}/${BEHAVIOR_DIR}/behavior_complexity_summary.json"
      fi
      echo "seed_done seed=${seed} gpu=${gpu_id} date=$(date -Is)"
    done
    echo "worker_done idx=${worker_idx} gpu=${gpu_id} date=$(date -Is)"
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
  echo "scenebal1_router_reaudit_failed date=$(date -Is)" >&2
  exit "${rc}"
fi

RUNS=()
for seed in "${SEEDS[@]}"; do
  RUNS+=("reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${SOURCE_DATE_TAG}")
done

MACRO_DIR="reports/aggregate/${AGG_LABEL}_macro_${DATE_TAG}"
RAW_MACRO_DIR="reports/aggregate/${AGG_LABEL}_raw_macro_${DATE_TAG}"
OLD_DIR="reports/aggregate/${AGG_LABEL}_oldclaim_${DATE_TAG}"
DECISION_JSON="reports/aggregate/${AGG_LABEL}_decision_audit_${DATE_TAG}.json"
DECISION_MD="reports/aggregate/${AGG_LABEL}_decision_audit_${DATE_TAG}.md"

"${PY}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --router-eval-dir "${EVAL_DIR}" \
  --behavior-dir "${BEHAVIOR_DIR}" \
  --out-dir "${MACRO_DIR}" \
  2>&1 | tee "logs/scenebal1_router_conf05_collect_macro_${SEED_LABEL}_${DATE_TAG}.log"

"${PY}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --router-eval-dir "${EVAL_DIR}" \
  --behavior-dir "${BEHAVIOR_DIR}" \
  --macro-score-column oracle_loss_macro_subtype_event \
  --out-dir "${RAW_MACRO_DIR}" \
  2>&1 | tee "logs/scenebal1_router_conf05_collect_raw_macro_${SEED_LABEL}_${DATE_TAG}.log"

"${PY}" scripts/73_v31_collect_oldclaim_gate.py \
  --runs "${RUNS[@]}" \
  --metrics-eval-dir "${EVAL_DIR}" \
  --behavior-dir "${BEHAVIOR_DIR}" \
  --behavior-eval-dir "${EVAL_DIR}" \
  --out-dir "${OLD_DIR}" \
  2>&1 | tee "logs/scenebal1_router_conf05_collect_oldclaim_${SEED_LABEL}_${DATE_TAG}.log"

"${PY}" scripts/75_v31_decide_scenebal1_stress_claim.py \
  --oldclaim-dir "${OLD_DIR}" \
  --macro-dir "${MACRO_DIR}" \
  --raw-macro-dir "${RAW_MACRO_DIR}" \
  --expected-seeds "${#SEEDS[@]}" \
  --label "${DECISION_LABEL}" \
  --out-json "${DECISION_JSON}" \
  --out-md "${DECISION_MD}" \
  2>&1 | tee "logs/scenebal1_router_conf05_decision_${SEED_LABEL}_${DATE_TAG}.log"

echo "scenebal1_router_reaudit_done date=$(date -Is) oldclaim_dir=${OLD_DIR} decision=${DECISION_JSON}"
