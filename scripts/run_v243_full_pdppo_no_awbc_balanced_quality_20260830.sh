#!/usr/bin/env bash
set -euo pipefail

# Clean structural ablation of V242: retain forecast reward, online context,
# hard feasibility masking, and subtype auxiliary prediction, while removing
# the training-only AWBC/BC teacher that may collapse the flexible action set.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=("${@:-3401 3402 3403 3404 3405}")
LOG_PREFIX="${LOG_PREFIX:-v243_no_awbc_pdppo}"

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="${RUN_PREFIX:-v243_no_awbc_pdppo_balanced_quality_dev}"
    export TOTAL_TIMESTEPS=100000 TRUTH_STEPS=36000 LOOKBACK=20
    export BUDGET=1.85 STARTUP_BUDGET=2.25 BUDGET_LABEL=b1p85
    export SENSOR_CFG="configs/sensors/windblown_sensors_physical_groups_v1.yaml"
    export EVENT_COVERAGE=0.55 MIN_DURATION=20 MAX_DURATION=64 MIN_GAP=12 LEAD_STEPS=8
    export EVENT_MICROSTRUCTURE_SIGMA=0.12 EVENT_MICROSTRUCTURE_ALPHA=0.15
    export EVENT_PARTICLE_MICROSTRUCTURE_CORRELATION=0.0 EVENT_SUBTYPE_ASSIGNMENT=stratified_duration
    export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY=0.0
    export EVENT_SUBTYPE_LATENT_ALPHA=0.15 EVENT_SUBTYPE_TARGET_LAG_STEPS=4
    export EVENT_SUBTYPE_CONTEXT_LEAD_STEPS=12 EVENT_SUBTYPE_CONTEXT_NOISE_STD=0.02 EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH=1.0
    export PARTICLE_LATENT_DIAMETER_SCALE=0.28 PARTICLE_LATENT_VELOCITY_SCALE=4.8 FLUX_LATENT_SIGMA=2.0 THERMAL_LATENT_SURFACE_SCALE=2.4
    export NOWCAST_LEAD_STEPS=8 NOWCAST_WIND_NOISE_STD=1.4 NOWCAST_HUMIDITY_NOISE_STD=4.2 NOWCAST_TEMPERATURE_NOISE_STD=1.0 NOWCAST_SOLAR_NOISE_STD=35.0
    export AGENT_CONTEXT_COLUMNS="${AGENT_CONTEXT_COLUMNS:-agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c agent_context_nowcast_solar_radiation_wm2 agent_context_quality_forecast_gmx500_weather_station agent_context_quality_forecast_lps10_pyranometer agent_context_quality_forecast_si111_surface_ir agent_context_quality_forecast_parsivel2_disdrometer agent_context_quality_forecast_flowcapt_fc4}"
    export INCLUDE_ALERT_CONTEXT_FEATURES=1 CONTEXT_FEATURE_DIM=9 CONTEXT_FUSION_MODE=gated_add
    export CHANNEL_QUALITY_ENABLED=1 CHANNEL_QUALITY_MODE=condition_dependent_crossover_balanced
    export CHANNEL_QUALITY_SENSOR_IDS="gmx500_weather_station lps10_pyranometer si111_surface_ir parsivel2_disdrometer flowcapt_fc4"
    export SENSOR_QUALITY_COLUMNS="agent_context_quality_gmx500_weather_station agent_context_quality_lps10_pyranometer agent_context_quality_si111_surface_ir agent_context_quality_parsivel2_disdrometer agent_context_quality_flowcapt_fc4"
    export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER=6.0 SENSOR_QUALITY_AVAILABILITY_FLOOR=0.1 CHANNEL_QUALITY_DEGRADED_COVERAGE=0.0 CHANNEL_QUALITY_DEGRADED_VALUE=0.1
    export CHANNEL_QUALITY_MIN_DURATION_STEPS=12 CHANNEL_QUALITY_MAX_DURATION_STEPS=48 CHANNEL_QUALITY_MIN_GAP_STEPS=12 CHANNEL_QUALITY_REPORT_NOISE_STD=0.02
    export QUALITY_CONTEXT_ACTION_SCORE="${QUALITY_CONTEXT_ACTION_SCORE:-0}"
    export AWBC_COEF=0 BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0 ORACLE_SUBTYPE_TEACHER_REPEAT=0 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=8
    export SUBTYPE_AUX_COEF=0.3 SUBTYPE_LOSS_WEIGHTING=1 SUBTYPE_ACTION_CE_COEF=0 SUBTYPE_ACTION_EVENT_ONLY=0 EXCLUDE_SUBTYPE_LATENTS_FROM_STATE=1
    export TRAINABLE_ACTION_PRIOR=0 NONLINEAR_ACTION_EMBEDDING=1 FORECAST_VALUE_HEAD=0 REWARD_LOSS_NORMALIZATION=none REWARD_PROXY_MODE=forecast
    export TARGET_WEIGHTS="1 1 1 1 1 1 1 1 1" COMMON_RANDOM_NUMBERS=0 SEPARATE_ACTOR_CRITIC_GRAD_CLIP=1
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/${LOG_PREFIX}_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "${SEEDS[$i]}" "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
