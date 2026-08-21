#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260622}"
SEEDS_TEXT="${SEEDS_TEXT:-141 142 143 144 145 146}"
GPU_IDS_TEXT="${ACTIONAUX_GPU_IDS:-0 1 2 3 4 5}"
ROUTER_CONF="${ROUTER_CONF:-0.5}"
EVAL_DIR="${EVAL_DIR:-eval_router_conf05_actionaux_${DATE_TAG}}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-behavior_audit_router_conf05_actionaux_${DATE_TAG}}"

VARIANT="weaker_latent10_actionaux_ce05_margin005"
RUN_PREFIX="v31_robust_weakerlatent10_actionaux_finalbenchmark"
BUDGET_LABEL="h075fb_latent10actaux"

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

echo "actionaux_pilot_start date=$(date -Is) seeds=${SEEDS[*]} variant=${VARIANT} router_conf=${ROUTER_CONF}"

configure_actionaux_env() {
  # Same moderate subtype-separation stress as the failed weaker_latent10
  # robustness row, plus a small action-level subtype auxiliary objective.
  export SUBTYPE_LATENT_ALPHA=0.58
  export PARTICLE_LATENT_DIAMETER_SCALE_MM=0.56
  export PARTICLE_LATENT_VELOCITY_SCALE_MS=7.4
  export FLUX_LATENT_LINEAR_SCALE=0.00125
  export FLUX_LATENT_LINEAR_CLIP=2.2
  export THERMAL_LATENT_SURFACE_SCALE_C=5.8
  export SUBTYPE_ACTION_CE_COEF=0.5
  export SUBTYPE_ACTION_MARGIN_COEF=0.05
  export SUBTYPE_ACTION_MARGIN=0.25
}

run_seed() {
  local seed="$1"
  local gpu_id="$2"
  local log_file="logs/actionaux_${VARIANT}_seed${seed}_${DATE_TAG}.log"

  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export SKIP_COLLECT=1
    export DEVICE="${DEVICE:-cuda}"
    export ORACLE_DEVICE="${ORACLE_DEVICE:-auto}"
    export ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
    export EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-cpu}"
    export RUN_PREFIX="${RUN_PREFIX}"
    export BUDGET_LABEL="${BUDGET_LABEL}"
    export DATE_TAG="${DATE_TAG}"
    configure_actionaux_env

    echo "actionaux_seed_start variant=${VARIANT} seed=${seed} gpu=${gpu_id} date=$(date -Is)"
    bash scripts/run_v31_scenebal2_seed_sweep_20260621.sh "${seed}"
    echo "actionaux_seed_done variant=${VARIANT} seed=${seed} gpu=${gpu_id} date=$(date -Is)"
  ) >"${log_file}" 2>&1
}

echo "actionaux_train_start variant=${VARIANT} date=$(date -Is)"
pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu_id="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  run_seed "${seed}" "${gpu_id}" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done
if [[ "${rc}" -ne 0 ]]; then
  echo "actionaux_train_failed variant=${VARIANT} date=$(date -Is)" >&2
  exit "${rc}"
fi
echo "actionaux_train_done variant=${VARIANT} date=$(date -Is)"

AGG_LABEL="actionaux_${VARIANT}_${SEED_LABEL}"
echo "actionaux_collect_start variant=${VARIANT} date=$(date -Is)"
SCENEBAL_ROUTER_GPU_IDS="${GPU_IDS_TEXT}" \
RUN_PREFIX="${RUN_PREFIX}" \
BUDGET_LABEL="${BUDGET_LABEL}" \
SOURCE_DATE_TAG="${DATE_TAG}" \
ROUTER_CONF="${ROUTER_CONF}" \
EVAL_DIR="${EVAL_DIR}" \
BEHAVIOR_DIR="${BEHAVIOR_DIR}" \
AGG_LABEL="${AGG_LABEL}" \
DECISION_LABEL="Action-Aux Weaker-Latent Pilot ${SEED_LABEL}" \
DATE_TAG="${DATE_TAG}" \
PY="${PY}" \
bash scripts/run_v31_scenebal1_router_conf05_reaudit_20260621.sh "${SEEDS[@]}"
echo "actionaux_collect_done variant=${VARIANT} date=$(date -Is)"

"${PY}" scripts/79_v31_collect_robustness.py \
  --label "Action-Aux Weaker-Latent Pilot ${SEED_LABEL}" \
  --entries "${VARIANT}=reports/aggregate/${AGG_LABEL}_decision_audit_${DATE_TAG}.json" \
  --out-dir "reports/aggregate/actionaux_pilot_${SEED_LABEL}_${DATE_TAG}"

echo "actionaux_pilot_done date=$(date -Is) out_dir=reports/aggregate/actionaux_pilot_${SEED_LABEL}_${DATE_TAG}"
