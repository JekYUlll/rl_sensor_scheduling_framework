#!/usr/bin/env bash
set -euo pipefail

# Clean scene-side forecast-horizon screen. The legal online state remains a
# noisy meteorological forecast, but its eight-step lead matches the l8 dynamic
# value horizon measured by the receding diagnostic. Forecast error is increased
# relative to the four-step setting; no teacher, rule action, or event label is
# added to PD-PPO.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2441 2442 2443}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v225_nowcast_horizon_aligned_dev}"
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
export NOWCAST_LEAD_STEPS=8
export NOWCAST_WIND_NOISE_STD=1.4
export NOWCAST_HUMIDITY_NOISE_STD=4.2
export NOWCAST_TEMPERATURE_NOISE_STD=1.0

bash scripts/run_v137_generic_physical_scene_gate_20260826.sh scene
bash scripts/run_v137_generic_physical_scene_gate_20260826.sh receding
