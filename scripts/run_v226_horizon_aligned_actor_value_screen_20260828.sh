#!/usr/bin/env bash
set -euo pipefail

# Final horizon-aligned clean screen. The policy sees the same noisy eight-step
# weather forecast as V225, while direct actor-logit pretraining and on-policy
# forecast-value targets use the matching l8 horizon. All targets are generated
# only on the policy-training partition by the frozen forecaster; no event
# label, hand-crafted action, or bandit information enters execution.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2451 2452 2453}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v226_horizon_aligned_actor_value_dev}"
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
export FORECAST_VALUE_HEAD=0
export FORECAST_VALUE_AUX_COEF=0.20
export FORECAST_VALUE_AUX_STRIDE=32
export FORECAST_VALUE_AUX_LOOKAHEAD_STEPS=8
export FORECAST_VALUE_AUX_LOSS=smooth_l1
export FORECAST_VALUE_RANKING_COEF=0.10
export GREEDY_LOOKAHEAD_STEPS=8
export BC_PRETRAIN_STEPS_OVERRIDE=4096
export BC_PRETRAIN_EPOCHS_OVERRIDE=4
export BC_PRETRAIN_LOSS_COEF_OVERRIDE=1.0
export BC_PRETRAIN_TARGET_MODE=forecast_value_regression

bash scripts/run_v137_generic_physical_scene_gate_20260826.sh scene
