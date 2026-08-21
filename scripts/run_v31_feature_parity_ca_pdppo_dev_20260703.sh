#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260703capdppo}"
LOG_DIR="${LOG_DIR:-logs/ca_pdppo_dev_${DATE_TAG}}"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2 3 4 5}"
BOOTSTRAP_DRAWS="${BOOTSTRAP_DRAWS:-10000}"
GREEDY_MAX_STEPS="${GREEDY_MAX_STEPS:--1}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224)
fi
read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_TEXT"
mkdir -p "$LOG_DIR"

echo "[ca-pdppo-dev] date_tag=${DATE_TAG} seeds=${SEEDS[*]} gpus=${GPU_IDS_ARRAY[*]}"

run_training_variant() {
  local variant="$1"
  local run_prefix="$2"
  local budget_label="$3"
  local include_alert="$4"
  local context_encoder="$5"
  local context_hidden_dim="$6"
  local n_workers="${#GPU_IDS_ARRAY[@]}"
  local pids=()

  echo "[ca-pdppo-dev] training variant=${variant} run_prefix=${run_prefix}"
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
      export SUBTYPE_ROUTER_ENABLED=0
      export EVENT_AWARE_CRITIC=0
      export EVENT_GATED_ACTOR=0
      export INCLUDE_EVENT_FLAG_IN_STATE=0
      export INCLUDE_ALERT_CONTEXT_FEATURES="$include_alert"
      export CONTEXT_ENCODER="$context_encoder"
      export CONTEXT_FEATURE_DIM=20
      export CONTEXT_HIDDEN_DIM="$context_hidden_dim"
      export ALERT_CONTEXT_THRESHOLD="${ALERT_CONTEXT_THRESHOLD:-0.5}"
      export ALERT_CONTEXT_TREND_LOOKBACK="${ALERT_CONTEXT_TREND_LOOKBACK:-6}"
      export ENT_COEF="${ENT_COEF:-0.0075}"
      export N_STEPS="${N_STEPS:-1024}"
      export N_EPOCHS="${N_EPOCHS:-8}"
      export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
      echo "[ca-pdppo-dev] worker=${worker_idx} gpu=${CUDA_VISIBLE_DEVICES} variant=${variant} seeds=${worker_seeds[*]}"
      bash scripts/run_v31_scenebal2_seed_sweep_20260621.sh "${worker_seeds[@]}"
    ) > "${LOG_DIR}/${variant}_worker${worker_idx}.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "[ca-pdppo-dev] training done variant=${variant}"
}

run_framework_baselines() {
  local variant="$1"
  local run_prefix="$2"
  local budget_label="$3"
  local out_root="reports/aggregate/contextaware_pdppo_${variant}_dev_${DATE_TAG}"
  echo "[ca-pdppo-dev] replay baselines variant=${variant} out_root=${out_root}"
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

run_training_variant "original_clean" "v31_pdppo_original_clean_scenebal2_dev" "h075origclean" 0 0 64
run_framework_baselines "original_clean" "v31_pdppo_original_clean_scenebal2_dev" "h075origclean"

run_training_variant "feature_parity" "v31_pdppo_feature_parity_scenebal2_dev" "h075featpar" 1 0 64
run_framework_baselines "feature_parity" "v31_pdppo_feature_parity_scenebal2_dev" "h075featpar"

run_training_variant "ca_pdppo" "v31_pdppo_ca_context_scenebal2_dev" "h075capdppo" 1 1 64
run_framework_baselines "ca_pdppo" "v31_pdppo_ca_context_scenebal2_dev" "h075capdppo"

"$PY" scripts/82_v31_collect_contextaware_pdppo_dev.py \
  --variant "original_clean=reports/aggregate/contextaware_pdppo_original_clean_dev_${DATE_TAG}" \
  --variant "feature_parity=reports/aggregate/contextaware_pdppo_feature_parity_dev_${DATE_TAG}" \
  --variant "ca_pdppo=reports/aggregate/contextaware_pdppo_ca_pdppo_dev_${DATE_TAG}" \
  --out-dir "reports/aggregate/contextaware_pdppo_dev_${DATE_TAG}" \
  --bootstrap-draws "$BOOTSTRAP_DRAWS" \
  2>&1 | tee "${LOG_DIR}/contextaware_pdppo_collect.log"

echo "[ca-pdppo-dev] complete reports/aggregate/contextaware_pdppo_dev_${DATE_TAG}"
