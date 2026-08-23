#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_batch() {
  local seeds="$1"
  SEEDS_OVERRIDE="$seeds" \
  RUN_PREFIX_OVERRIDE=v120_full_intensity_context_gate_dev \
  CONTEXT_OUT_OVERRIDE=reports/aggregate/v120_full_intensity_context_gate_20260823 \
  EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH_OVERRIDE=1.0 \
  bash scripts/run_v103_frequency_cost_scene_gate_20260823.sh scene

  SEEDS_OVERRIDE="$seeds" \
  GPU_IDS="0 1 2 3 4 5" \
  RUN_PREFIX_OVERRIDE=v134_fresh_soft_onpolicy_temp075 \
  FORECAST_VALUE_AUX_LOSS_OVERRIDE=soft_ce \
  FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.75 \
  bash scripts/run_v125_dense_onpolicy_forecast_value_pilot_20260823.sh
}

run_batch "1407 1408 1409 1410 1411 1412"
run_batch "1413 1414 1415 1416 1417 1418"
run_batch "1419 1420 1421 1422 1423 1424"
