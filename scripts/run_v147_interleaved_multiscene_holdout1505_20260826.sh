#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

train_prefix="${TRAIN_PREFIX_OVERRIDE:-v147_interleaved_multiscene_train}"
holdout_prefix="${HOLDOUT_PREFIX_OVERRIDE:-v147_interleaved_multiscene_holdout}"
training_sources="reports/v138_generic_physical_statefix_gate_dev_seed1502_b1p75_20260822 reports/v138_generic_physical_statefix_gate_dev_seed1503_b1p75_20260822 reports/v138_generic_physical_statefix_gate_dev_seed1504_b1p75_20260822"
common_env=(
  GPU_IDS="${GPU_IDS:-0}"
  FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.25
  FORECAST_VALUE_AUX_COEF_OVERRIDE=1.0
  FORECAST_VALUE_AUX_STRIDE_OVERRIDE=16
  FORECAST_VALUE_AUX_LOSS_OVERRIDE=mse
  ENT_COEF_OVERRIDE=0.005
  FORECAST_VALUE_HEAD=1
  FORECAST_VALUE_HEAD_SCALE=1.0
  FORECAST_VALUE_HEAD_HIDDEN_DIM=128
  FORECAST_VALUE_HEAD_MODE=independent
)

env \
  "${common_env[@]}" \
  SEEDS_OVERRIDE=1501 \
  RUN_PREFIX_OVERRIDE="$train_prefix" \
  TOTAL_TIMESTEPS_OVERRIDE="${TOTAL_TIMESTEPS_OVERRIDE:-81920}" \
  CHECKPOINT_SELECTION_INTERVAL_UPDATES_OVERRIDE="${CHECKPOINT_SELECTION_INTERVAL_UPDATES_OVERRIDE:-0}" \
  TRAINING_CONTROL_SOURCE_RUN_DIRS_OVERRIDE="$training_sources" \
  bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh

checkpoint="reports/${train_prefix}_seed1501_b1p75_20260822/custom_ppo.pt"
env \
  "${common_env[@]}" \
  SEEDS_OVERRIDE=1505 \
  RUN_PREFIX_OVERRIDE="$holdout_prefix" \
  POLICY_CHECKPOINT_SOURCE_OVERRIDE="$checkpoint" \
  TRAINING_CONTROL_SOURCE_RUN_DIRS_OVERRIDE="" \
  bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh
