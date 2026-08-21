#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260621}"
PILOT_TAG="${PILOT_TAG:-behaviorbrg2_pilot_83_92_${DATE_TAG}}"
SEEDS=("$@")
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  SEEDS=(83 84 86 87 91 92)
fi

mkdir -p logs

echo "behaviorbrg2_parallel_start date=$(date -Is) seeds=${SEEDS[*]} pilot=${PILOT_TAG}"

pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu="${BRG2_GPU_IDS:-${BRG_GPU_IDS:-0 1 2 3 4 5}}"
  # shellcheck disable=SC2206
  gpu_ids=($gpu)
  gpu_id="${gpu_ids[$((idx % ${#gpu_ids[@]}))]}"
  log_file="logs/behaviorbrg2_pilot_seed${seed}_${DATE_TAG}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export SKIP_COLLECT=1
    export DEVICE="${DEVICE:-cuda}"
    export ORACLE_DEVICE="${ORACLE_DEVICE:-auto}"
    export ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
    export EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-cpu}"
    echo "behaviorbrg2_seed_start seed=${seed} gpu=${gpu_id} date=$(date -Is)"
    bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg2_seed_sweep_20260621.sh "${seed}"
    echo "behaviorbrg2_seed_done seed=${seed} gpu=${gpu_id} date=$(date -Is)"
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
  echo "behaviorbrg2_parallel_failed date=$(date -Is)" >&2
  exit "${rc}"
fi

RUNS=()
for seed in "${SEEDS[@]}"; do
  RUNS+=("reports/v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg2_seed${seed}_h075ctxolbrg2_${DATE_TAG}")
done

MACRO_DIR="reports/aggregate/behaviorbrg2_pilot_83_92_macro_${DATE_TAG}"
RAW_MACRO_DIR="reports/aggregate/behaviorbrg2_pilot_83_92_raw_macro_${DATE_TAG}"
OLD_DIR="reports/aggregate/behaviorbrg2_pilot_83_92_oldclaim_${DATE_TAG}"

"${PY}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --out-dir "${MACRO_DIR}" \
  2>&1 | tee "logs/behaviorbrg2_collect_macro_83_92_${DATE_TAG}.log"

"${PY}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --router-eval-dir . \
  --out-dir "${RAW_MACRO_DIR}" \
  2>&1 | tee "logs/behaviorbrg2_collect_raw_macro_83_92_${DATE_TAG}.log"

"${PY}" scripts/73_v31_collect_oldclaim_gate.py \
  --runs "${RUNS[@]}" \
  --out-dir "${OLD_DIR}" \
  2>&1 | tee "logs/behaviorbrg2_collect_oldclaim_83_92_${DATE_TAG}.log"

"${PY}" scripts/74_v31_write_balancedobjective_report.py \
  --macro-dir "${MACRO_DIR}" \
  --raw-macro-dir "${RAW_MACRO_DIR}" \
  --oldclaim-dir "${OLD_DIR}" \
  --out-file "${OLD_DIR}/BREAKTHROUGH_REPORT.md" \
  --title "BRG-2 Matched-Router Observable-Regime-Belief PPO Pilot Report" \
  --notes "BRG-2 bounded follow-up: raw/eval subtype-router confidence matched at 0.70, entropy coefficient 0.0075, seeds ${SEEDS[*]}." \
  2>&1 | tee "logs/behaviorbrg2_write_report_83_92_${DATE_TAG}.log"

echo "behaviorbrg2_parallel_done date=$(date -Is)"
