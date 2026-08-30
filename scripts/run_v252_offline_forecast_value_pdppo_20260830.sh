#!/usr/bin/env bash
set -euo pipefail

# V252: bounded test of one-time forecast-value pretraining. Candidate costs
# are collected once before PPO; no oracle candidate evaluation is performed
# during on-policy rollout collection.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/.conda/envs/darts/bin/python}"
if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(4201 4202)
fi

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-v252_offline_forecast_value_pdppo_sixch_dev_retry}"
    export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS_OVERRIDE:-50000}" TRUTH_STEPS=36000 LOOKBACK=20
    export BUDGET=1.75 STARTUP_BUDGET=2.15 BUDGET_LABEL=b1p75
    export SENSOR_CFG="configs/sensors/windblown_sensors_flexible_subset_v6_physical_channels.yaml"
    export EVENT_COVERAGE=0.55 MIN_DURATION=20 MAX_DURATION=64 MIN_GAP=12 LEAD_STEPS=8
    export EVENT_MICROSTRUCTURE_SIGMA=0.12 EVENT_MICROSTRUCTURE_ALPHA=0.15
    export EVENT_PARTICLE_MICROSTRUCTURE_CORRELATION=0.0 EVENT_SUBTYPE_ASSIGNMENT=stratified
    export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY=0.0
    export EVENT_SUBTYPE_LATENT_ALPHA=0.15 EVENT_SUBTYPE_TARGET_LAG_STEPS=4
    export EVENT_SUBTYPE_CONTEXT_LEAD_STEPS=12 EVENT_SUBTYPE_CONTEXT_NOISE_STD=0.02
    export EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH=1.0
    export PARTICLE_LATENT_DIAMETER_SCALE=0.28 PARTICLE_LATENT_VELOCITY_SCALE=4.8
    export FLUX_LATENT_SIGMA=2.0 THERMAL_LATENT_SURFACE_SCALE=2.4
    export NOWCAST_LEAD_STEPS=8 NOWCAST_WIND_NOISE_STD=1.4
    export NOWCAST_HUMIDITY_NOISE_STD=4.2 NOWCAST_TEMPERATURE_NOISE_STD=1.0 NOWCAST_SOLAR_NOISE_STD=35.0
    export AGENT_CONTEXT_COLUMNS="" INCLUDE_ALERT_CONTEXT_FEATURES=1 CONTEXT_FEATURE_DIM=20 CONTEXT_FUSION_MODE=gated_add
    export CHANNEL_QUALITY_ENABLED=1 CHANNEL_QUALITY_MODE=condition_dependent_crossover_balanced
    export CHANNEL_QUALITY_SENSOR_IDS="met_station_core radiometer_basic shielded_thermo_hygro surface_temp_ir laser_disdrometer fc4_flux"
    export SENSOR_QUALITY_COLUMNS="agent_context_quality_met_station_core agent_context_quality_radiometer_basic agent_context_quality_shielded_thermo_hygro agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer agent_context_quality_fc4_flux"
    export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER=6.0 SENSOR_QUALITY_AVAILABILITY_FLOOR=0.1
    export CHANNEL_QUALITY_DEGRADED_COVERAGE=0.0 CHANNEL_QUALITY_DEGRADED_VALUE=0.1
    export CHANNEL_QUALITY_MIN_DURATION_STEPS=12 CHANNEL_QUALITY_MAX_DURATION_STEPS=48
    export CHANNEL_QUALITY_MIN_GAP_STEPS=12 CHANNEL_QUALITY_REPORT_NOISE_STD=0.02
    export TARGET_WEIGHTS="1 1 1 1 1 3 1 1 1" COMMON_RANDOM_NUMBERS=0
    export EXCLUDE_SUBTYPE_LATENTS_FROM_STATE=1

    # One-time offline forecast-value targets; no per-rollout oracle calls.
    export BC_PRETRAIN_STEPS="${BC_PRETRAIN_STEPS_OVERRIDE:-512}"
    export BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS_OVERRIDE:-10}"
    export BC_PRETRAIN_LOSS_COEF=1.0
    export BC_PRETRAIN_TARGET_MODE="${BC_PRETRAIN_TARGET_MODE_OVERRIDE:-forecast_value_regression}"
    export BC_SOFT_TEMPERATURE="${BC_SOFT_TEMPERATURE_OVERRIDE:-1.0}" FORECAST_VALUE_HEAD=0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_AUX_STRIDE=16 FORECAST_VALUE_AUX_LOOKAHEAD_STEPS=8
    export AWBC_COEF=0 AWBC_DECAY_TIMESTEPS=0 AWBC_EVENT_ONLY=0 AWBC_LABEL_STRIDE=4 AWBC_TEACHER_MODE=oracle_greedy
    export ORACLE_SUBTYPE_TEACHER_REPEAT=0 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=8
    export SUBTYPE_AUX_COEF=0.3 SUBTYPE_LOSS_WEIGHTING=1 SUBTYPE_ACTION_CE_COEF=0
    export TRAINABLE_ACTION_PRIOR=0 NONLINEAR_ACTION_EMBEDDING=1
    export ENT_COEF=0.02 CHANNEL_MARGINAL_ENTROPY_COEF=0
    export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS_OVERRIDE:-50000}"
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=0
    export EVALUATION_POLICY_MODE=deterministic
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/v252_offline_forecast_value_pdppo_sixch_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "${SEEDS[$i]}" "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
