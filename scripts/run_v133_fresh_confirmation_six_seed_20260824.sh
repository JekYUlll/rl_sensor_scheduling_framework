#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seeds="1401 1402 1403 1404 1405 1406"

# Generate frozen scene/evaluator assets without inspecting final policy output.
SEEDS_OVERRIDE="$seeds" \
RUN_PREFIX_OVERRIDE=v120_full_intensity_context_gate_dev \
CONTEXT_OUT_OVERRIDE=reports/aggregate/v120_full_intensity_context_gate_20260823 \
EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH_OVERRIDE=1.0 \
bash scripts/run_v103_frequency_cost_scene_gate_20260823.sh scene

# Train and evaluate the frozen V132 method once on every fresh scene.
SEEDS_OVERRIDE="$seeds" \
GPU_IDS="0 1 2 3 4 5" \
RUN_PREFIX_OVERRIDE=v133_fresh_soft_onpolicy_temp075 \
FORECAST_VALUE_AUX_LOSS_OVERRIDE=soft_ce \
FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE=0.75 \
bash scripts/run_v125_dense_onpolicy_forecast_value_pilot_20260823.sh
