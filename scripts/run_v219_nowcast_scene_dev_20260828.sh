#!/usr/bin/env bash
set -euo pipefail

# Development-only nowcast scene. The scheduler receives only label-free
# meteorological nowcast columns; synthetic event-alert tails stay out of state.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2401 2402 2403 2404 2405}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v219_nowcast_scene_dev}"
export SENSOR_CFG_OVERRIDE="${SENSOR_CFG_OVERRIDE:-configs/sensors/windblown_sensors_flexible_subset_v7_coverage_balanced.yaml}"
export BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-1.85}"
export STARTUP_BUDGET_OVERRIDE="${STARTUP_BUDGET_OVERRIDE:-2.25}"
export BUDGET_LABEL_OVERRIDE="${BUDGET_LABEL_OVERRIDE:-b1p85}"
export EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE=stratified_duration
export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY_OVERRIDE=0.0
export CHANNEL_QUALITY_ENABLED_OVERRIDE=1
export CHANNEL_QUALITY_MODE_OVERRIDE=independent
export AGENT_CONTEXT_COLUMNS="agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c"
# The actor splits context from the trailing state dimensions.  This development
# configuration supplies exactly the three label-free meteorological nowcasts.
export CONTEXT_FEATURE_DIM=3
export INCLUDE_ALERT_CONTEXT_FEATURES=0
export NOWCAST_LEAD_STEPS=4
export NOWCAST_WIND_NOISE_STD=1.0
export NOWCAST_HUMIDITY_NOISE_STD=3.0
export NOWCAST_TEMPERATURE_NOISE_STD=0.7

bash scripts/run_v137_generic_physical_scene_gate_20260826.sh scene
bash scripts/run_v137_generic_physical_scene_gate_20260826.sh receding
