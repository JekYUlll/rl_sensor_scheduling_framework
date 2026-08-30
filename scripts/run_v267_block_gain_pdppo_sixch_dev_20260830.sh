#!/usr/bin/env bash
set -euo pipefail

# V267 tests an action-sensitive block objective.  The reward is the
# six-step forecast-loss reduction relative to holding the previously executed
# subset; no bandit action, teacher action, or test label is used.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/.conda/envs/darts/bin/python}"
if [[ "$#" -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(5701 5702); fi

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="${RUN_PREFIX:-v267_block_gain_pdppo_sixch_dev}"
    export TOTAL_TIMESTEPS=50000 TRUTH_STEPS=36000 LOOKBACK=20 FORECAST_HORIZON=6
    export BUDGET=1.75 STARTUP_BUDGET=2.15 BUDGET_LABEL=b1p75
    export SENSOR_CFG="configs/sensors/windblown_sensors_flexible_subset_v6_physical_channels.yaml"
    export EVENT_COVERAGE=0.55 MIN_DURATION=20 MAX_DURATION=64 MIN_GAP=12 LEAD_STEPS=8
    export EVENT_MICROSTRUCTURE_SIGMA=0.12 EVENT_MICROSTRUCTURE_ALPHA=0.15
    export EVENT_PARTICLE_MICROSTRUCTURE_CORRELATION=0.0 EVENT_SUBTYPE_ASSIGNMENT=stratified
    export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY=0.0
    export EVENT_SUBTYPE_LATENT_ALPHA=0.15 EVENT_SUBTYPE_TARGET_LAG_STEPS=4
    export EVENT_SUBTYPE_CONTEXT_LEAD_STEPS=12 EVENT_SUBTYPE_CONTEXT_NOISE_STD=0.02 EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH=1.0
    export PARTICLE_LATENT_DIAMETER_SCALE=0.28 PARTICLE_LATENT_VELOCITY_SCALE=4.8 FLUX_LATENT_SIGMA=2.0 THERMAL_LATENT_SURFACE_SCALE=2.4
    export NOWCAST_LEAD_STEPS=8 NOWCAST_WIND_NOISE_STD=1.4 NOWCAST_HUMIDITY_NOISE_STD=4.2 NOWCAST_TEMPERATURE_NOISE_STD=1.0 NOWCAST_SOLAR_NOISE_STD=35.0
    export AGENT_CONTEXT_COLUMNS="agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert" INCLUDE_ALERT_CONTEXT_FEATURES=1 CONTEXT_FEATURE_DIM=20 CONTEXT_FUSION_MODE=gated_add
    export CHANNEL_QUALITY_ENABLED=1 CHANNEL_QUALITY_MODE=condition_dependent_crossover_balanced
    export CHANNEL_QUALITY_SENSOR_IDS="met_station_core radiometer_basic shielded_thermo_hygro surface_temp_ir laser_disdrometer fc4_flux"
    export SENSOR_QUALITY_COLUMNS="agent_context_quality_met_station_core agent_context_quality_radiometer_basic agent_context_quality_shielded_thermo_hygro agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer agent_context_quality_fc4_flux"
    export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER=6.0 SENSOR_QUALITY_AVAILABILITY_FLOOR=0.1 CHANNEL_QUALITY_DEGRADED_COVERAGE=0.0 CHANNEL_QUALITY_DEGRADED_VALUE=0.1
    export CHANNEL_QUALITY_MIN_DURATION_STEPS=12 CHANNEL_QUALITY_MAX_DURATION_STEPS=48 CHANNEL_QUALITY_MIN_GAP_STEPS=12 CHANNEL_QUALITY_REPORT_NOISE_STD=0.02
    export AWBC_COEF=0 BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0
    export ORACLE_SUBTYPE_TEACHER_REPEAT=0 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=6 SUBTYPE_AUX_COEF=0.3 SUBTYPE_LOSS_WEIGHTING=1 SUBTYPE_ACTION_CE_COEF=0
    export SUBTYPE_ACTION_EVENT_ONLY=0 EXCLUDE_SUBTYPE_LATENTS_FROM_STATE=1 TRAINABLE_ACTION_PRIOR=0 NONLINEAR_ACTION_EMBEDDING=1
    export REWARD_LOSS_NORMALIZATION=none REWARD_PROXY_MODE="${REWARD_PROXY_MODE:-forecast_block_gain}"
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_AUX_STRIDE=32 FORECAST_VALUE_AUX_LOOKAHEAD_STEPS=6
    export TARGET_WEIGHTS="1 1 1 1 1 3 1 1 1" COMMON_RANDOM_NUMBERS=0 TEMPORAL_ENCODER=1 TEMPORAL_HIDDEN_DIM=64 SEPARATE_ACTOR_CRITIC_GRAD_CLIP=1
    export GAMMA=0.99 GAE_LAMBDA=0.95 \
      DECISION_ONLY_POLICY_UPDATES="${DECISION_ONLY_POLICY_UPDATES:-0}" \
      DECISION_BLOCK_CREDIT="${DECISION_BLOCK_CREDIT:-0}" \
      DECISION_BLOCK_REWARD_MODE="${DECISION_BLOCK_REWARD_MODE:-sum}"
    export EVENT_START_PROB="${EVENT_START_PROB:-0.70}"
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/${LOG_PREFIX:-v267_block_gain_pdppo_sixch}_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "${SEEDS[$i]}" "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
