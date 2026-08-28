#!/usr/bin/env bash
set -euo pipefail

# Diagnostic-only gate for the verified five-instrument physical grouping.
# Any PPO artifact from the minimal pipeline invocation is ignored: admission is
# decided exclusively from fresh static selection, action geometry, and l8
# receding diagnostics.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2461 2462 2463 2464 2465}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v227_physical_groups_scene_dev}"
export SENSOR_CFG_OVERRIDE="${SENSOR_CFG_OVERRIDE:-configs/sensors/windblown_sensors_physical_groups_v1.yaml}"
export BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-1.85}"
export STARTUP_BUDGET_OVERRIDE="${STARTUP_BUDGET_OVERRIDE:-2.25}"
export BUDGET_LABEL_OVERRIDE="${BUDGET_LABEL_OVERRIDE:-b1p85}"
export TOTAL_TIMESTEPS_OVERRIDE=1024
export EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE=stratified_duration
export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY_OVERRIDE=0.0
export CHANNEL_QUALITY_ENABLED_OVERRIDE=0
export INCLUDE_ALERT_CONTEXT_FEATURES=0
export CONTEXT_FEATURE_DIM_OVERRIDE=0
export TEACHER_CALM_SENSORS="gmx500_weather_station lps10_pyranometer"
export TEACHER_PARTICLE_SENSORS="gmx500_weather_station parsivel2_disdrometer"
export TEACHER_FLUX_SENSORS="gmx500_weather_station flowcapt_fc4"
export TEACHER_THERMAL_SENSORS="si111_surface_ir lps10_pyranometer"

bash scripts/run_v137_generic_physical_scene_gate_20260826.sh scene
bash scripts/run_v137_generic_physical_scene_gate_20260826.sh receding
