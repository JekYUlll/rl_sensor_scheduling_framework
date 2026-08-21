#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260621}"
SEEDS=("$@")
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  SEEDS=(83 84 86 87 91 92)
fi
if [[ -z "${SEED_LABEL:-}" ]]; then
  if [[ "${SEEDS[*]}" == "83 84 86 87 91 92" ]]; then
    SEED_LABEL="83_92"
  else
    SEED_LABEL="$(IFS=_; echo "${SEEDS[*]}")"
  fi
fi
PILOT_TAG="${PILOT_TAG:-temporal1_pilot_${SEED_LABEL}_${DATE_TAG}}"

mkdir -p logs

echo "temporal1_parallel_start date=$(date -Is) seeds=${SEEDS[*]} pilot=${PILOT_TAG}"

pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu="${TEMPORAL1_GPU_IDS:-${BRG3_GPU_IDS:-${BRG2_GPU_IDS:-${BRG_GPU_IDS:-0 1 2 3 4 5}}}}"
  # shellcheck disable=SC2206
  gpu_ids=($gpu)
  gpu_id="${gpu_ids[$((idx % ${#gpu_ids[@]}))]}"
  log_file="logs/temporal1_pilot_seed${seed}_${DATE_TAG}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export SKIP_COLLECT=1
    export DEVICE="${DEVICE:-cuda}"
    export ORACLE_DEVICE="${ORACLE_DEVICE:-auto}"
    export ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
    export EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-cpu}"
    echo "temporal1_seed_start seed=${seed} gpu=${gpu_id} date=$(date -Is)"
    bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_temporal1_seed_sweep_20260621.sh "${seed}"
    echo "temporal1_seed_done seed=${seed} gpu=${gpu_id} date=$(date -Is)"
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
  echo "temporal1_parallel_failed date=$(date -Is)" >&2
  exit "${rc}"
fi

RUNS=()
for seed in "${SEEDS[@]}"; do
  RUNS+=("reports/v31_metpair_backbone_context_ortholinear_balancedobjective_temporal1_seed${seed}_h075ctxoltemp1_${DATE_TAG}")
done

MACRO_DIR="reports/aggregate/temporal1_pilot_${SEED_LABEL}_macro_${DATE_TAG}"
RAW_MACRO_DIR="reports/aggregate/temporal1_pilot_${SEED_LABEL}_raw_macro_${DATE_TAG}"
OLD_DIR="reports/aggregate/temporal1_pilot_${SEED_LABEL}_oldclaim_${DATE_TAG}"

"${PY}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --out-dir "${MACRO_DIR}" \
  2>&1 | tee "logs/temporal1_collect_macro_${SEED_LABEL}_${DATE_TAG}.log"

"${PY}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --router-eval-dir . \
  --out-dir "${RAW_MACRO_DIR}" \
  2>&1 | tee "logs/temporal1_collect_raw_macro_${SEED_LABEL}_${DATE_TAG}.log"

"${PY}" scripts/73_v31_collect_oldclaim_gate.py \
  --runs "${RUNS[@]}" \
  --out-dir "${OLD_DIR}" \
  2>&1 | tee "logs/temporal1_collect_oldclaim_${SEED_LABEL}_${DATE_TAG}.log"

"${PY}" scripts/74_v31_write_balancedobjective_report.py \
  --macro-dir "${MACRO_DIR}" \
  --raw-macro-dir "${RAW_MACRO_DIR}" \
  --oldclaim-dir "${OLD_DIR}" \
  --out-file "${OLD_DIR}/BREAKTHROUGH_REPORT.md" \
  --title "TEMPORAL-1 Lead-Aware Observable-Regime-Belief PPO Pilot Report" \
  --notes "TEMPORAL-1 bounded pivot: longer subtype context/teacher/auxiliary lookahead, no action CE/margin, seeds ${SEEDS[*]}." \
  2>&1 | tee "logs/temporal1_write_report_${SEED_LABEL}_${DATE_TAG}.log"

echo "temporal1_parallel_done date=$(date -Is)"
