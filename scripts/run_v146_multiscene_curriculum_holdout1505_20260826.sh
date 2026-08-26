#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

prefix="${RUN_PREFIX_OVERRIDE:-v146_multiscene_curriculum}"
previous=""
common_env=(
  GPU_IDS="${GPU_IDS:-0}"
  RUN_PREFIX_OVERRIDE="$prefix"
  FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE="${FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE:-0.25}"
  FORECAST_VALUE_AUX_COEF_OVERRIDE="${FORECAST_VALUE_AUX_COEF_OVERRIDE:-1.0}"
  FORECAST_VALUE_AUX_STRIDE_OVERRIDE="${FORECAST_VALUE_AUX_STRIDE_OVERRIDE:-16}"
  FORECAST_VALUE_AUX_LOSS_OVERRIDE="${FORECAST_VALUE_AUX_LOSS_OVERRIDE:-mse}"
  ENT_COEF_OVERRIDE="${ENT_COEF_OVERRIDE:-0.005}"
  FORECAST_VALUE_HEAD=1
  FORECAST_VALUE_HEAD_SCALE="${FORECAST_VALUE_HEAD_SCALE:-1.0}"
  FORECAST_VALUE_HEAD_HIDDEN_DIM="${FORECAST_VALUE_HEAD_HIDDEN_DIM:-128}"
  FORECAST_VALUE_HEAD_MODE=independent
)

# Curriculum stages use only each scene's policy-training partition for
# optimization and its calibration/validation partition for checkpoint choice.
for seed in 1501 1502 1503 1504; do
  env \
    "${common_env[@]}" \
    SEEDS_OVERRIDE="$seed" \
    POLICY_INIT_SOURCE_OVERRIDE="$previous" \
    POLICY_CHECKPOINT_SOURCE_OVERRIDE="" \
    bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh
  previous="reports/${prefix}_seed${seed}_b1p75_20260822/custom_ppo.pt"
done

# Freeze the final curriculum checkpoint before evaluating the held-out scene.
env \
  "${common_env[@]}" \
  SEEDS_OVERRIDE=1505 \
  POLICY_INIT_SOURCE_OVERRIDE="" \
  POLICY_CHECKPOINT_SOURCE_OVERRIDE="$previous" \
  bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh
