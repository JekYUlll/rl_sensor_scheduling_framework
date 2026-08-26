#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seeds=(1501 1502 1503 1504 1505)
pids=()
for index in "${!seeds[@]}"; do
  holdout="${seeds[$index]}"
  gpu="$index"
  train_seed=""
  sources=()
  for seed in "${seeds[@]}"; do
    [[ "$seed" == "$holdout" ]] && continue
    if [[ -z "$train_seed" ]]; then
      train_seed="$seed"
    else
      sources+=("reports/v138_generic_physical_statefix_gate_dev_seed${seed}_b1p75_20260822")
    fi
  done
  (
    common=(
      GPU_IDS="$gpu"
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
    train_prefix="v151_loo_train_holdout${holdout}"
    env "${common[@]}" \
      SEEDS_OVERRIDE="$train_seed" \
      RUN_PREFIX_OVERRIDE="$train_prefix" \
      LOG_DIR_OVERRIDE="logs/${train_prefix}" \
      TOTAL_TIMESTEPS_OVERRIDE=81920 \
      CHECKPOINT_SELECTION_INTERVAL_UPDATES_OVERRIDE=5 \
      CHECKPOINT_REQUIRE_VALID_BEHAVIOR_OVERRIDE=1 \
      TRAINING_CONTROL_SOURCE_RUN_DIRS_OVERRIDE="${sources[*]}" \
      bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh

    checkpoint="reports/${train_prefix}_seed${train_seed}_b1p75_20260822/custom_ppo.pt"
    env "${common[@]}" \
      SEEDS_OVERRIDE="$holdout" \
      RUN_PREFIX_OVERRIDE="v151_loo_holdout${holdout}" \
      LOG_DIR_OVERRIDE="logs/v151_loo_holdout${holdout}" \
      POLICY_CHECKPOINT_SOURCE_OVERRIDE="$checkpoint" \
      TRAINING_CONTROL_SOURCE_RUN_DIRS_OVERRIDE="" \
      bash scripts/run_v139_generic_physical_pdppo_dev_20260826.sh
  ) >"logs/v151_loo_holdout${holdout}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
