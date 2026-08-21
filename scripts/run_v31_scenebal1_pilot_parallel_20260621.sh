#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260621}"
SEEDS=("$@")
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  SEEDS=(93 94 95 96 97 98)
fi
if [[ -z "${SEED_LABEL:-}" ]]; then
  SEED_LABEL="$(IFS=_; echo "${SEEDS[*]}")"
fi
PILOT_TAG="${PILOT_TAG:-scenebal1_pilot_${SEED_LABEL}_${DATE_TAG}}"

mkdir -p logs

echo "scenebal1_parallel_start date=$(date -Is) seeds=${SEEDS[*]} pilot=${PILOT_TAG}"

pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu="${SCENEBAL1_GPU_IDS:-${TEMPORAL1_GPU_IDS:-0 1 2 3 4 5}}"
  # shellcheck disable=SC2206
  gpu_ids=($gpu)
  gpu_id="${gpu_ids[$((idx % ${#gpu_ids[@]}))]}"
  log_file="logs/scenebal1_pilot_seed${seed}_${DATE_TAG}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export SKIP_COLLECT=1
    export DEVICE="${DEVICE:-cuda}"
    export ORACLE_DEVICE="${ORACLE_DEVICE:-auto}"
    export ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
    export EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-cpu}"
    echo "scenebal1_seed_start seed=${seed} gpu=${gpu_id} date=$(date -Is)"
    bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal1_seed_sweep_20260621.sh "${seed}"
    echo "scenebal1_seed_done seed=${seed} gpu=${gpu_id} date=$(date -Is)"
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
  echo "scenebal1_parallel_failed date=$(date -Is)" >&2
  exit "${rc}"
fi

RUNS=()
for seed in "${SEEDS[@]}"; do
  RUNS+=("reports/v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal1_seed${seed}_h075ctxolscbal1_${DATE_TAG}")
done

MACRO_DIR="reports/aggregate/scenebal1_pilot_${SEED_LABEL}_macro_${DATE_TAG}"
RAW_MACRO_DIR="reports/aggregate/scenebal1_pilot_${SEED_LABEL}_raw_macro_${DATE_TAG}"
OLD_DIR="reports/aggregate/scenebal1_pilot_${SEED_LABEL}_oldclaim_${DATE_TAG}"

"${PY}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --out-dir "${MACRO_DIR}" \
  2>&1 | tee "logs/scenebal1_collect_macro_${SEED_LABEL}_${DATE_TAG}.log"

"${PY}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --router-eval-dir . \
  --out-dir "${RAW_MACRO_DIR}" \
  2>&1 | tee "logs/scenebal1_collect_raw_macro_${SEED_LABEL}_${DATE_TAG}.log"

"${PY}" scripts/73_v31_collect_oldclaim_gate.py \
  --runs "${RUNS[@]}" \
  --out-dir "${OLD_DIR}" \
  2>&1 | tee "logs/scenebal1_collect_oldclaim_${SEED_LABEL}_${DATE_TAG}.log"

"${PY}" scripts/74_v31_write_balancedobjective_report.py \
  --macro-dir "${MACRO_DIR}" \
  --raw-macro-dir "${RAW_MACRO_DIR}" \
  --oldclaim-dir "${OLD_DIR}" \
  --out-file "${OLD_DIR}/SCENEBAL1_REPORT.md" \
  --title "SCENEBAL-1 Balanced-Scene PPO Pilot Report" \
  --notes "SCENEBAL-1 bounded pivot: TEMPORAL-1 PPO/teacher/lead mechanism retained; simulator/target weights rebalanced to break the met+fc4 raw-step shortcut; seeds ${SEEDS[*]}." \
  2>&1 | tee "logs/scenebal1_write_report_${SEED_LABEL}_${DATE_TAG}.log"

echo "scenebal1_parallel_done date=$(date -Is)"

