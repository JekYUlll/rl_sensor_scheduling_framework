#!/usr/bin/env bash
set -euo pipefail

# Development-only learnability test after the V219 minimal-training closure.
# The scheduler receives a normalized, label-free four-step meteorological
# nowcast and no synthetic alert, event-label, or quality-tail features.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2411 2412 2413 2414 2415}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v221_nowcast_normalized_dev}"
export SENSOR_CFG_OVERRIDE="${SENSOR_CFG_OVERRIDE:-configs/sensors/windblown_sensors_flexible_subset_v7_coverage_balanced.yaml}"
export BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-1.85}"
export STARTUP_BUDGET_OVERRIDE="${STARTUP_BUDGET_OVERRIDE:-2.25}"
export BUDGET_LABEL_OVERRIDE="${BUDGET_LABEL_OVERRIDE:-b1p85}"
export TOTAL_TIMESTEPS_OVERRIDE="${TOTAL_TIMESTEPS_OVERRIDE:-50000}"
export EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE=stratified_duration
export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY_OVERRIDE=0.0
export CHANNEL_QUALITY_ENABLED_OVERRIDE=0
export AGENT_CONTEXT_COLUMNS="agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c"
export CONTEXT_FEATURE_DIM_OVERRIDE=3
export INCLUDE_ALERT_CONTEXT_FEATURES=0
export NOWCAST_LEAD_STEPS=4
export NOWCAST_WIND_NOISE_STD=1.0
export NOWCAST_HUMIDITY_NOISE_STD=3.0
export NOWCAST_TEMPERATURE_NOISE_STD=0.7

bash scripts/run_v137_generic_physical_scene_gate_20260826.sh scene
bash scripts/run_v137_generic_physical_scene_gate_20260826.sh receding
