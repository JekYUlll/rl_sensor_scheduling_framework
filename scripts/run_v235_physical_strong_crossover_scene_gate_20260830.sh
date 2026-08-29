#!/usr/bin/env bash
set -euo pipefail

# Development-only admission screen. It retains V234's physical groups,
# fixed effective loads, budget, evaluator, and arbitrary-subset geometry while
# strengthening only the prespecified weather-conditioned reliability profiles.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2701 2702 2703 2704 2705}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v235_physical_strong_crossover_dev}"
export CONTEXT_OUT_OVERRIDE="${CONTEXT_OUT_OVERRIDE:-reports/aggregate/v235_physical_strong_crossover_context_20260830}"
export OUT_ROOT="${OUT_ROOT:-reports/aggregate/v235_physical_strong_crossover_gate_20260830}"
export SENSOR_CFG_OVERRIDE="${SENSOR_CFG_OVERRIDE:-configs/sensors/windblown_sensors_physical_groups_v1.yaml}"
export BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-1.85}"
export STARTUP_BUDGET_OVERRIDE="${STARTUP_BUDGET_OVERRIDE:-2.25}"
export BUDGET_LABEL_OVERRIDE="${BUDGET_LABEL_OVERRIDE:-b1p85}"
export TOTAL_TIMESTEPS_OVERRIDE="${TOTAL_TIMESTEPS_OVERRIDE:-1024}"
export EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE=stratified_duration
export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY_OVERRIDE=0.0
export CHANNEL_QUALITY_ENABLED_OVERRIDE=1
export CHANNEL_QUALITY_MODE_OVERRIDE=condition_dependent_crossover_strong
export CHANNEL_QUALITY_DEGRADED_COVERAGE_OVERRIDE=0.0
export CHANNEL_QUALITY_DEGRADED_VALUE_OVERRIDE=0.10
export CHANNEL_QUALITY_REPORT_NOISE_STD_OVERRIDE=0.02
export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER_OVERRIDE=6.0
export SENSOR_QUALITY_AVAILABILITY_FLOOR_OVERRIDE=0.10
export CHANNEL_QUALITY_SENSOR_IDS="gmx500_weather_station lps10_pyranometer si111_surface_ir parsivel2_disdrometer flowcapt_fc4"
export SENSOR_QUALITY_COLUMNS="agent_context_quality_gmx500_weather_station agent_context_quality_lps10_pyranometer agent_context_quality_si111_surface_ir agent_context_quality_parsivel2_disdrometer agent_context_quality_flowcapt_fc4"
export AGENT_CONTEXT_COLUMNS="agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c"
export CONTEXT_FEATURE_DIM_OVERRIDE=8
export INCLUDE_ALERT_CONTEXT_FEATURES=0
export NOWCAST_LEAD_STEPS=8
export NOWCAST_WIND_NOISE_STD=1.4
export NOWCAST_HUMIDITY_NOISE_STD=4.2
export NOWCAST_TEMPERATURE_NOISE_STD=1.0

exec bash scripts/run_v232_physical_weather_quality_scene_gate_20260828.sh "${1:-all}"
