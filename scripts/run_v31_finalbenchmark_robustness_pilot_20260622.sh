#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260622}"
SEEDS_TEXT="${SEEDS_TEXT:-141 142 143 144 145 146}"
GPU_IDS_TEXT="${ROBUSTNESS_GPU_IDS:-0 1 2 3 4 5}"
ROUTER_CONF="${ROUTER_CONF:-0.5}"
EVAL_DIR="${EVAL_DIR:-eval_router_conf05_robustness_${DATE_TAG}}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-behavior_audit_router_conf05_robustness_${DATE_TAG}}"
ROBUSTNESS_VARIANTS_TEXT="${ROBUSTNESS_VARIANTS_TEXT:-event_mix_flux30 weaker_latent10}"

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

echo "robustness_pilot_start date=$(date -Is) seeds=${SEEDS[*]} router_conf=${ROUTER_CONF}"

variant_run_prefix() {
  case "$1" in
    event_mix_flux30)
      echo "v31_robust_eventmix_flux30_finalbenchmark"
      ;;
    event_mix_flux10)
      echo "v31_robust_eventmix_flux10_finalbenchmark"
      ;;
    event_mix_uniform)
      echo "v31_robust_eventmix_uniform_finalbenchmark"
      ;;
    weaker_latent10)
      echo "v31_robust_weakerlatent10_finalbenchmark"
      ;;
    *)
      echo "Unknown robustness variant: $1" >&2
      return 2
      ;;
  esac
}

variant_budget_label() {
  case "$1" in
    event_mix_flux30)
      echo "h075fb_mixf30"
      ;;
    event_mix_flux10)
      echo "h075fb_mixf10"
      ;;
    event_mix_uniform)
      echo "h075fb_mixuniform"
      ;;
    weaker_latent10)
      echo "h075fb_latent10"
      ;;
    *)
      echo "Unknown robustness variant: $1" >&2
      return 2
      ;;
  esac
}

configure_variant_env() {
  local variant="$1"
  case "$variant" in
    event_mix_flux30)
      # Small event-mix shift: the specialist bottleneck is unchanged, but the
      # flux subtype is less rare than in the main benchmark.
      export PARTICLE_PROB=0.35
      export FLUX_PROB=0.30
      export THERMAL_PROB=0.35
      ;;
    event_mix_flux10)
      # Stronger event-mix stress: flux events are rare, while the particle and
      # thermal specialists dominate the mixture.
      export PARTICLE_PROB=0.45
      export FLUX_PROB=0.10
      export THERMAL_PROB=0.45
      ;;
    event_mix_uniform)
      # Balanced subtype mix with no dominant event type.
      export PARTICLE_PROB=0.34
      export FLUX_PROB=0.33
      export THERMAL_PROB=0.33
      ;;
    weaker_latent10)
      # Moderate subtype-separation stress. Keep the same subtype mix and
      # sensor geometry while weakening the latent cues by roughly 10%.
      export SUBTYPE_LATENT_ALPHA=0.58
      export PARTICLE_LATENT_DIAMETER_SCALE_MM=0.56
      export PARTICLE_LATENT_VELOCITY_SCALE_MS=7.4
      export FLUX_LATENT_LINEAR_SCALE=0.00125
      export FLUX_LATENT_LINEAR_CLIP=2.2
      export THERMAL_LATENT_SURFACE_SCALE_C=5.8
      ;;
    *)
      echo "Unknown robustness variant: ${variant}" >&2
      exit 2
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
  local log_file="logs/robustness_${variant}_seed${seed}_${DATE_TAG}.log"

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
    configure_variant_env "${variant}"

    echo "robustness_seed_start variant=${variant} seed=${seed} gpu=${gpu_id} date=$(date -Is)"
    bash scripts/run_v31_scenebal2_seed_sweep_20260621.sh "${seed}"
    echo "robustness_seed_done variant=${variant} seed=${seed} gpu=${gpu_id} date=$(date -Is)"
  ) >"${log_file}" 2>&1
}

run_train_variant() {
  local variant="$1"
  echo "robustness_variant_train_start variant=${variant} date=$(date -Is)"
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
    echo "robustness_variant_train_failed variant=${variant} date=$(date -Is)" >&2
    exit "${rc}"
  fi
  echo "robustness_variant_train_done variant=${variant} date=$(date -Is)"
}

collect_variant() {
  local label="$1"
  local run_prefix="$2"
  local budget_label="$3"
  local agg_label="robustness_${label}_${SEED_LABEL}"

  echo "robustness_collect_start label=${label} date=$(date -Is)"
  SCENEBAL_ROUTER_GPU_IDS="${GPU_IDS_TEXT}" \
  RUN_PREFIX="${run_prefix}" \
  BUDGET_LABEL="${budget_label}" \
  SOURCE_DATE_TAG="${DATE_TAG}" \
  ROUTER_CONF="${ROUTER_CONF}" \
  EVAL_DIR="${EVAL_DIR}" \
  BEHAVIOR_DIR="${BEHAVIOR_DIR}" \
  AGG_LABEL="${agg_label}" \
  DECISION_LABEL="Final-Benchmark Robustness ${label} ${SEED_LABEL}" \
  DATE_TAG="${DATE_TAG}" \
  PY="${PY}" \
  bash scripts/run_v31_scenebal1_router_conf05_reaudit_20260621.sh "${SEEDS[@]}"
  echo "robustness_collect_done label=${label} date=$(date -Is)"
}

# shellcheck disable=SC2206
ROBUSTNESS_VARIANTS=(${ROBUSTNESS_VARIANTS_TEXT})
ENTRY_ARGS=()

for variant in "${ROBUSTNESS_VARIANTS[@]}"; do
  run_train_variant "${variant}"
  collect_variant "${variant}" "$(variant_run_prefix "${variant}")" "$(variant_budget_label "${variant}")"
  ENTRY_ARGS+=("${variant}=reports/aggregate/robustness_${variant}_${SEED_LABEL}_decision_audit_${DATE_TAG}.json")
done

"${PY}" scripts/79_v31_collect_robustness.py \
  --label "Final-Benchmark Robustness Pilot ${SEED_LABEL}" \
  --entries "${ENTRY_ARGS[@]}" \
  --out-dir "reports/aggregate/robustness_pilot_${SEED_LABEL}_${DATE_TAG}"

echo "robustness_pilot_done date=$(date -Is) out_dir=reports/aggregate/robustness_pilot_${SEED_LABEL}_${DATE_TAG}"
