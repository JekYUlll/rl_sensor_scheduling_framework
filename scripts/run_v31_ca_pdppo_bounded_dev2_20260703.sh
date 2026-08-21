#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260703capdppodev2}"
REF_DATE_TAG="${REF_DATE_TAG:-20260703capdppo}"
LOG_DIR="${LOG_DIR:-logs/ca_pdppo_bounded_dev2_${DATE_TAG}}"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2 3 4}"
BOOTSTRAP_DRAWS="${BOOTSTRAP_DRAWS:-10000}"
GREEDY_MAX_STEPS="${GREEDY_MAX_STEPS:--1}"
SWITCH_LIMIT="${SWITCH_LIMIT:-0.00662}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224)
fi
read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_TEXT"
mkdir -p "$LOG_DIR"

echo "[ca-pdppo-dev2] date_tag=${DATE_TAG} seeds=${SEEDS[*]} gpus=${GPU_IDS_ARRAY[*]}"

run_training_variant() {
  local variant="$1"
  local run_prefix="$2"
  local budget_label="$3"
  local context_hidden_dim="$4"
  local context_fusion_mode="$5"
  local context_layer_norm="$6"
  local n_steps="$7"
  local learning_rate="$8"
  local ent_coef="$9"

  if [[ "$SKIP_TRAINING" == "1" ]]; then
    echo "[ca-pdppo-dev2] skip training variant=${variant}"
    return 0
  fi

  local n_workers="${#GPU_IDS_ARRAY[@]}"
  local pids=()

  echo "[ca-pdppo-dev2] training variant=${variant} run_prefix=${run_prefix}"
  for worker_idx in "${!GPU_IDS_ARRAY[@]}"; do
    (
      worker_seeds=()
      for seed_idx in "${!SEEDS[@]}"; do
        if (( seed_idx % n_workers == worker_idx )); then
          worker_seeds+=("${SEEDS[$seed_idx]}")
        fi
      done
      if [[ "${#worker_seeds[@]}" -eq 0 ]]; then
        exit 0
      fi

      export CUDA_VISIBLE_DEVICES="${GPU_IDS_ARRAY[$worker_idx]}"
      export PY="$PY"
      export DEVICE="${DEVICE:-cuda}"
      export ORACLE_DEVICE="${ORACLE_DEVICE:-cuda}"
      export ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
      export EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-cpu}"
      export RUN_PREFIX="$run_prefix"
      export BUDGET_LABEL="$budget_label"
      export DATE_TAG="$DATE_TAG"
      export SKIP_COLLECT=1
      export SKIP_ROUTER_EVAL=1
      export SKIP_REPLAY_GATE=1
      export SKIP_BEHAVIOR_AUDIT=1

      # Clean CA-PD-PPO identity: online alert features + context encoder,
      # forecast-loss reward, and feasibility-masked PPO. No bandit-dependent
      # priors, labels, residual actions, or bandit-margin rewards are used.
      export SUBTYPE_ROUTER_ENABLED=0
      export EVENT_AWARE_CRITIC=0
      export EVENT_GATED_ACTOR=0
      export INCLUDE_EVENT_FLAG_IN_STATE=0
      export INCLUDE_ALERT_CONTEXT_FEATURES=1
      export CONTEXT_ENCODER=1
      export CONTEXT_FEATURE_DIM=20
      export CONTEXT_HIDDEN_DIM="$context_hidden_dim"
      export CONTEXT_FUSION_MODE="$context_fusion_mode"
      export CONTEXT_LAYER_NORM="$context_layer_norm"
      export ALERT_CONTEXT_THRESHOLD="${ALERT_CONTEXT_THRESHOLD:-0.5}"
      export ALERT_CONTEXT_TREND_LOOKBACK="${ALERT_CONTEXT_TREND_LOOKBACK:-6}"

      export ENT_COEF="$ent_coef"
      export N_STEPS="$n_steps"
      export N_EPOCHS="${N_EPOCHS:-8}"
      export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
      export LEARNING_RATE="$learning_rate"

      echo "[ca-pdppo-dev2] worker=${worker_idx} gpu=${CUDA_VISIBLE_DEVICES} variant=${variant} seeds=${worker_seeds[*]}"
      bash scripts/run_v31_scenebal2_seed_sweep_20260621.sh "${worker_seeds[@]}"
    ) > "${LOG_DIR}/${variant}_worker${worker_idx}.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "[ca-pdppo-dev2] training done variant=${variant}"
}

run_framework_baselines() {
  local variant="$1"
  local run_prefix="$2"
  local budget_label="$3"
  local out_root="reports/aggregate/ca_pdppo_bounded_dev2_${variant}_${DATE_TAG}"

  if [[ "$SKIP_BASELINES" == "1" ]]; then
    echo "[ca-pdppo-dev2] skip framework baselines variant=${variant}"
    return 0
  fi

  echo "[ca-pdppo-dev2] replay baselines variant=${variant} out_root=${out_root}"
  "$PY" scripts/81_v31_framework_baseline_supplements.py \
    --run-glob "reports/${run_prefix}_seed*_${budget_label}_${DATE_TAG}" \
    --seeds "${SEEDS[@]}" \
    --out-root "$out_root" \
    --router-eval-dir . \
    --replay-dir replay_gate_explicit_static_noguard \
    --oracle-device "${EVAL_ORACLE_DEVICE:-cpu}" \
    --policies context_bandit forecast_greedy \
    --context-thresholds 0.5 \
    --greedy-max-steps "$GREEDY_MAX_STEPS" \
    2>&1 | tee "${LOG_DIR}/${variant}_framework_baselines.log"
}

collect_variants() {
  local collect_args=()
  local ref_root="reports/aggregate/contextaware_pdppo_ca_pdppo_dev_${REF_DATE_TAG}"
  if [[ -f "${ref_root}/framework_baseline_seed_metrics.csv" ]]; then
    collect_args+=(--variant "ca_current=${ref_root}")
  fi
  collect_args+=(--variant "ctx128=reports/aggregate/ca_pdppo_bounded_dev2_ctx128_${DATE_TAG}")
  collect_args+=(--variant "gated=reports/aggregate/ca_pdppo_bounded_dev2_gated_${DATE_TAG}")
  collect_args+=(--variant "gated_ctx128=reports/aggregate/ca_pdppo_bounded_dev2_gated_ctx128_${DATE_TAG}")
  collect_args+=(--variant "nsteps2048=reports/aggregate/ca_pdppo_bounded_dev2_nsteps2048_${DATE_TAG}")

  "$PY" scripts/84_v31_collect_ca_pdppo_bounded_dev2.py \
    "${collect_args[@]}" \
    --out-dir "reports/aggregate/ca_pdppo_bounded_dev2_${DATE_TAG}" \
    --bootstrap-draws "$BOOTSTRAP_DRAWS" \
    --switch-limit "$SWITCH_LIMIT" \
    2>&1 | tee "${LOG_DIR}/ca_pdppo_bounded_dev2_collect.log"
}

run_training_variant "ctx128" "v31_pdppo_ca_ctx128_scenebal2_dev2" "h075cactx128" 128 concat 0 1024 3e-4 0.0075
run_framework_baselines "ctx128" "v31_pdppo_ca_ctx128_scenebal2_dev2" "h075cactx128"

run_training_variant "gated" "v31_pdppo_ca_gated_scenebal2_dev2" "h075cagated" 64 gated_add 1 1024 3e-4 0.0075
run_framework_baselines "gated" "v31_pdppo_ca_gated_scenebal2_dev2" "h075cagated"

run_training_variant "gated_ctx128" "v31_pdppo_ca_gated_ctx128_scenebal2_dev2" "h075cagat128" 128 gated_add 1 1024 3e-4 0.0075
run_framework_baselines "gated_ctx128" "v31_pdppo_ca_gated_ctx128_scenebal2_dev2" "h075cagat128"

run_training_variant "nsteps2048" "v31_pdppo_ca_nsteps2048_scenebal2_dev2" "h075can2048" 64 concat 0 2048 3e-4 0.0075
run_framework_baselines "nsteps2048" "v31_pdppo_ca_nsteps2048_scenebal2_dev2" "h075can2048"

collect_variants

echo "[ca-pdppo-dev2] complete reports/aggregate/ca_pdppo_bounded_dev2_${DATE_TAG}"
