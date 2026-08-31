#!/usr/bin/env bash
set -euo pipefail

# V346 tests candidate forecast-value supervision on states actually visited
# by the policy. It retains the forecast-loss reward, hard feasibility mask,
# arbitrary subset geometry, and direct state--candidate-mask representation.
# No bandit signal, comparator action, or final-test label is used.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=(6871 6872)
POLICY_SEEDS=(${POLICY_SEEDS_OVERRIDE:-7381 7382})
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local idx="$1" seed="${SEEDS[$1]}" policy_seed="${POLICY_SEEDS[$1]}"
  (
    export CUDA_VISIBLE_DEVICES="$((idx + GPU_OFFSET))"
    export RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-v346_onpolicy_forecast_aux_direct_mask}"
    export LOG_PREFIX="${LOG_PREFIX_OVERRIDE:-v346_onpolicy_forecast_aux_direct_mask}"
    export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS_OVERRIDE:-50000}" POLICY_SEED="$policy_seed"
    export CONTROL_SOURCE_RUN_DIR="reports/v338_recalibrated_scene_control_seed${seed}_b1p75_20260822"
    export EVENT_COVERAGE=0.70 MIN_DURATION=20 MAX_DURATION=64 MIN_GAP=12 LEAD_STEPS=8
    export EVENT_SUBTYPE_ASSIGNMENT=stratified
    export EVENT_SUBTYPE_PARTICLE_PROB=0.3333333333 EVENT_SUBTYPE_FLUX_PROB=0.3333333333 EVENT_SUBTYPE_THERMAL_PROB=0.3333333334
    export SENSOR_CFG="configs/sensors/windblown_sensors_flexible_subset_v6_physical_channels.yaml"
    export BUDGET=1.75 STARTUP_BUDGET=2.15 BUDGET_LABEL=b1p75
    export REWARD_PROXY_MODE=forecast REWARD_LOSS_NORMALIZATION=staticnorm_subtype EVENT_START_PROB=1.0
    export GREEDY_LOOKAHEAD_STEPS=6 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=6
    export DECISION_ONLY_POLICY_UPDATES=1 DECISION_BLOCK_CREDIT=0 DECISION_BLOCK_REWARD_MODE=sum
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 CHECKPOINT_SELECTION_MIN_UPDATE=5 CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=16384 BC_PRETRAIN_EPOCHS=20 BC_PRETRAIN_LOSS_COEF=1
    export BC_PRETRAIN_DECISION_ONLY=0 BC_PRETRAIN_TARGET_MODE=hard_forecast_value BC_SOFT_TEMPERATURE=1.0
    export FORECAST_VALUE_AUX_COEF="${FORECAST_VALUE_AUX_COEF:-0.10}" FORECAST_VALUE_AUX_STRIDE="${FORECAST_VALUE_AUX_STRIDE:-4}" FORECAST_VALUE_AUX_LOOKAHEAD_STEPS=6
    export FORECAST_VALUE_AUX_LOSS=mse FORECAST_VALUE_AUX_TEMPERATURE=1.0 FORECAST_VALUE_RANKING_COEF=0
    export CANDIDATE_INTERACTION_SCORE=0 DIRECT_MASK_ACTION_SCORE=1 FACTORIZED_ACTION_POLICY=0
    export AGENT_CONTEXT_COLUMNS="agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert" CONTEXT_FEATURE_DIM=20
    export TEMPORAL_ENCODER=1 TEMPORAL_HIDDEN_DIM=64 SEPARATE_ACTOR_CRITIC_GRAD_CLIP=1
    export ENT_COEF=0.02 EXCLUDE_SUBTYPE_LATENTS_FROM_STATE=1 NONLINEAR_ACTION_EMBEDDING=1
    export CHANNEL_QUALITY_ENABLED=1 CHANNEL_QUALITY_MODE=condition_dependent_crossover_balanced
    export CHANNEL_QUALITY_SENSOR_IDS="met_station_core radiometer_basic shielded_thermo_hygro surface_temp_ir laser_disdrometer fc4_flux"
    export SENSOR_QUALITY_COLUMNS="agent_context_quality_met_station_core agent_context_quality_radiometer_basic agent_context_quality_shielded_thermo_hygro agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer agent_context_quality_fc4_flux"
    export CHANNEL_QUALITY_MIN_DURATION_STEPS=12 CHANNEL_QUALITY_MAX_DURATION_STEPS=48 CHANNEL_QUALITY_MIN_GAP_STEPS=12
    export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER=6.0 SENSOR_QUALITY_AVAILABILITY_FLOOR=0.1 CHANNEL_QUALITY_DEGRADED_COVERAGE=0.0 CHANNEL_QUALITY_DEGRADED_VALUE=0.1 CHANNEL_QUALITY_REPORT_NOISE_STD=0.02
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v346_onpolicy_forecast_aux_direct_mask_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in 0 1; do run_one "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
