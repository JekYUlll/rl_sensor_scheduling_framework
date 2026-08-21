#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260621}"
SEEDS_TEXT="${SEEDS_TEXT:-117 118 119 120 121 122}"
GPU_IDS_TEXT="${MECH_ABLATION_GPU_IDS:-0 1 2 3 4 5}"
ROUTER_CONF="${ROUTER_CONF:-0.5}"
RUN_THRESHOLD_SENSITIVITY="${RUN_THRESHOLD_SENSITIVITY:-1}"

FULL_RUN_PREFIX="${FULL_RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2}"
FULL_BUDGET_LABEL="${FULL_BUDGET_LABEL:-h075ctxolscbal2}"
FULL_SOURCE_DATE_TAG="${FULL_SOURCE_DATE_TAG:-20260621}"

EVAL_DIR="${EVAL_DIR:-eval_router_conf05_mechablation_20260621}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-behavior_audit_router_conf05_mechablation_20260621}"

# shellcheck disable=SC2206
SEEDS=(${SEEDS_TEXT})
# shellcheck disable=SC2206
GPU_IDS=(${GPU_IDS_TEXT})

if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "No seeds configured" >&2
  exit 2
fi
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPU ids configured" >&2
  exit 2
fi

SEED_LABEL="$(IFS=_; echo "${SEEDS[*]}")"
mkdir -p logs reports/aggregate

echo "mechanism_ablation_start date=$(date -Is) seeds=${SEEDS[*]} router_conf=${ROUTER_CONF}"

variant_run_prefix() {
  case "$1" in
    no_imitation_guide)
      echo "v31_mechablate_no_imitation_guide_finalbenchmark"
      ;;
    no_regime_aux_path)
      echo "v31_mechablate_no_regime_aux_path_finalbenchmark"
      ;;
    no_staticnorm_train)
      echo "v31_mechablate_no_staticnorm_train_finalbenchmark"
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 2
      ;;
  esac
}

variant_budget_label() {
  case "$1" in
    no_imitation_guide)
      echo "h075fb_noimit"
      ;;
    no_regime_aux_path)
      echo "h075fb_noregaux"
      ;;
    no_staticnorm_train)
      echo "h075fb_nostatnormtrain"
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 2
      ;;
  esac
}

run_variant_seed() {
  local variant="$1"
  local seed="$2"
  local gpu_id="$3"
  local run_prefix
  local budget_label
  run_prefix="$(variant_run_prefix "$variant")"
  budget_label="$(variant_budget_label "$variant")"
  local log_file="logs/mechanism_ablation_${variant}_seed${seed}_${DATE_TAG}.log"

  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export SKIP_COLLECT=1
    export DEVICE="${DEVICE:-cuda}"
    export ORACLE_DEVICE="${ORACLE_DEVICE:-auto}"
    export ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
    export EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-cpu}"
    export RUN_PREFIX="${run_prefix}"
    export BUDGET_LABEL="${budget_label}"
    export DATE_TAG="${DATE_TAG}"

    case "$variant" in
      no_imitation_guide)
        export AWBC_COEF=0.0
        export BC_PRETRAIN_STEPS=0
        export BC_PRETRAIN_LOSS_COEF=0.0
        ;;
      no_regime_aux_path)
        export INCLUDE_OBSERVABLE_REGIME_BELIEF=0
        export EVENT_GATED_ACTOR=0
        export SUBTYPE_AUX_COEF=0.0
        ;;
      no_staticnorm_train)
        export REWARD_LOSS_NORMALIZATION=none
        ;;
      *)
        echo "Unknown variant: ${variant}" >&2
        exit 2
        ;;
    esac

    echo "mechanism_ablation_seed_start variant=${variant} seed=${seed} gpu=${gpu_id} date=$(date -Is)"
    bash scripts/run_v31_scenebal2_seed_sweep_20260621.sh "${seed}"
    echo "mechanism_ablation_seed_done variant=${variant} seed=${seed} gpu=${gpu_id} date=$(date -Is)"
  ) >"${log_file}" 2>&1
}

run_train_variant() {
  local variant="$1"
  echo "mechanism_ablation_variant_train_start variant=${variant} date=$(date -Is)"
  local pids=()
  for idx in "${!SEEDS[@]}"; do
    local seed="${SEEDS[$idx]}"
    local gpu_id="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
    run_variant_seed "${variant}" "${seed}" "${gpu_id}" &
    pids+=("$!")
  done

  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  if [[ "${rc}" -ne 0 ]]; then
    echo "mechanism_ablation_variant_train_failed variant=${variant} date=$(date -Is)" >&2
    exit "${rc}"
  fi
  echo "mechanism_ablation_variant_train_done variant=${variant} date=$(date -Is)"
}

collect_variant() {
  local label="$1"
  local run_prefix="$2"
  local budget_label="$3"
  local source_date_tag="$4"
  local agg_label="mechanism_ablation_${label}_${SEED_LABEL}"

  echo "mechanism_ablation_collect_start label=${label} date=$(date -Is)"
  SCENEBAL_ROUTER_GPU_IDS="${GPU_IDS_TEXT}" \
  RUN_PREFIX="${run_prefix}" \
  BUDGET_LABEL="${budget_label}" \
  SOURCE_DATE_TAG="${source_date_tag}" \
  ROUTER_CONF="${ROUTER_CONF}" \
  EVAL_DIR="${EVAL_DIR}" \
  BEHAVIOR_DIR="${BEHAVIOR_DIR}" \
  AGG_LABEL="${agg_label}" \
  DECISION_LABEL="Final-Benchmark Mechanism Ablation ${label} ${SEED_LABEL}" \
  DATE_TAG="${DATE_TAG}" \
  PY="${PY}" \
  bash scripts/run_v31_scenebal1_router_conf05_reaudit_20260621.sh "${SEEDS[@]}"
  echo "mechanism_ablation_collect_done label=${label} date=$(date -Is)"
}

collect_threshold_sensitivity() {
  local threshold="$1"
  local suffix
  suffix="$(printf '%s' "${threshold}" | tr '.' 'p')"
  local eval_dir="eval_router_conf${suffix}_sensitivity_${DATE_TAG}"
  local behavior_dir="behavior_audit_router_conf${suffix}_sensitivity_${DATE_TAG}"
  local agg_label="mechanism_ablation_threshold_conf${suffix}_${SEED_LABEL}"

  echo "mechanism_ablation_threshold_collect_start conf=${threshold} date=$(date -Is)"
  SCENEBAL_ROUTER_GPU_IDS="${GPU_IDS_TEXT}" \
  RUN_PREFIX="${FULL_RUN_PREFIX}" \
  BUDGET_LABEL="${FULL_BUDGET_LABEL}" \
  SOURCE_DATE_TAG="${FULL_SOURCE_DATE_TAG}" \
  ROUTER_CONF="${threshold}" \
  EVAL_DIR="${eval_dir}" \
  BEHAVIOR_DIR="${behavior_dir}" \
  AGG_LABEL="${agg_label}" \
  DECISION_LABEL="Final-Benchmark Threshold Sensitivity conf=${threshold} ${SEED_LABEL}" \
  DATE_TAG="${DATE_TAG}" \
  PY="${PY}" \
  bash scripts/run_v31_scenebal1_router_conf05_reaudit_20260621.sh "${SEEDS[@]}"
  echo "mechanism_ablation_threshold_collect_done conf=${threshold} date=$(date -Is)"
}

TRAIN_VARIANTS=(no_imitation_guide no_regime_aux_path no_staticnorm_train)

for variant in "${TRAIN_VARIANTS[@]}"; do
  run_train_variant "${variant}"
  collect_variant "${variant}" "$(variant_run_prefix "${variant}")" "$(variant_budget_label "${variant}")" "${DATE_TAG}"
done

collect_variant "full_reference" "${FULL_RUN_PREFIX}" "${FULL_BUDGET_LABEL}" "${FULL_SOURCE_DATE_TAG}"

if [[ "${RUN_THRESHOLD_SENSITIVITY}" == "1" || "${RUN_THRESHOLD_SENSITIVITY}" == "true" || "${RUN_THRESHOLD_SENSITIVITY}" == "yes" ]]; then
  collect_threshold_sensitivity "0.0"
  collect_threshold_sensitivity "0.7"
  collect_threshold_sensitivity "0.9"
fi

ENTRY_ARGS=(
  "full_reference=reports/aggregate/mechanism_ablation_full_reference_${SEED_LABEL}_decision_audit_${DATE_TAG}.json"
  "no_imitation_guide=reports/aggregate/mechanism_ablation_no_imitation_guide_${SEED_LABEL}_decision_audit_${DATE_TAG}.json"
  "no_regime_aux_path=reports/aggregate/mechanism_ablation_no_regime_aux_path_${SEED_LABEL}_decision_audit_${DATE_TAG}.json"
  "no_staticnorm_train=reports/aggregate/mechanism_ablation_no_staticnorm_train_${SEED_LABEL}_decision_audit_${DATE_TAG}.json"
)

if [[ "${RUN_THRESHOLD_SENSITIVITY}" == "1" || "${RUN_THRESHOLD_SENSITIVITY}" == "true" || "${RUN_THRESHOLD_SENSITIVITY}" == "yes" ]]; then
  ENTRY_ARGS+=(
    "threshold_conf0p0=reports/aggregate/mechanism_ablation_threshold_conf0p0_${SEED_LABEL}_decision_audit_${DATE_TAG}.json"
    "threshold_conf0p7=reports/aggregate/mechanism_ablation_threshold_conf0p7_${SEED_LABEL}_decision_audit_${DATE_TAG}.json"
    "threshold_conf0p9=reports/aggregate/mechanism_ablation_threshold_conf0p9_${SEED_LABEL}_decision_audit_${DATE_TAG}.json"
  )
fi

"${PY}" scripts/78_v31_collect_mechanism_ablation.py \
  --label "Final-Benchmark Mechanism Ablation Pilot ${SEED_LABEL}" \
  --reference full_reference \
  --entries "${ENTRY_ARGS[@]}" \
  --out-dir "reports/aggregate/mechanism_ablation_pilot_${SEED_LABEL}_${DATE_TAG}"

echo "mechanism_ablation_done date=$(date -Is) out_dir=reports/aggregate/mechanism_ablation_pilot_${SEED_LABEL}_${DATE_TAG}"
