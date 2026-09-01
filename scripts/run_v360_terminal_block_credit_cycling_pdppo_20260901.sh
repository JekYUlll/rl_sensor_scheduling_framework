#!/usr/bin/env bash
set -euo pipefail

# V360 tests terminal semi-Markov credit on the frozen V357 cycling scenes.
# Compared with V359, the only learning change is decision-block credit with
# the terminal reward of each six-step executable block. No bandit signal,
# privileged final-test label, or scene change is introduced.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=(${SCENE_SEEDS_OVERRIDE:-6901 6902 6903 6904})
POLICY_SEEDS=(${POLICY_SEEDS_OVERRIDE:-7471 7472 7473 7474})
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local idx="$1" seed="${SEEDS[$1]}" policy_seed="${POLICY_SEEDS[$1]}"
  (
    export CUDA_VISIBLE_DEVICES="$((idx + GPU_OFFSET))"
    export RUN_PREFIX="v360_terminal_block_credit_cycling_pdppo"
    export LOG_PREFIX="v360_terminal_block_credit_cycling_pdppo"
    export TOTAL_TIMESTEPS=50000 POLICY_SEED="$policy_seed"
    export CONTROL_SOURCE_RUN_DIR="reports/v357_confirmation_scene_control_seed${seed}_b1p75_20260822"
    export EVENT_COVERAGE=0.70 MIN_DURATION=20 MAX_DURATION=64 MIN_GAP=12 LEAD_STEPS=8
    export EVENT_SUBTYPE_ASSIGNMENT=cycling EVENT_SUBTYPE_CYCLE_STEPS=12
    export EVENT_SUBTYPE_PARTICLE_PROB=0.3333333333 EVENT_SUBTYPE_FLUX_PROB=0.3333333333 EVENT_SUBTYPE_THERMAL_PROB=0.3333333334
    export SENSOR_CFG="configs/sensors/windblown_sensors_flexible_subset_v6_physical_channels.yaml"
    export BUDGET=1.75 STARTUP_BUDGET=2.15 BUDGET_LABEL=b1p75
    export REWARD_PROXY_MODE=forecast REWARD_LOSS_NORMALIZATION=staticnorm_subtype EVENT_START_PROB=1.0
    export GREEDY_LOOKAHEAD_STEPS=6 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=6
    export DECISION_ONLY_POLICY_UPDATES=1 DECISION_BLOCK_CREDIT=1 DECISION_BLOCK_REWARD_MODE=terminal
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 CHECKPOINT_SELECTION_MIN_UPDATE=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_macro_subtype_event_staticnorm CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0
    export BC_PRETRAIN_DECISION_ONLY=0 BC_PRETRAIN_TARGET_MODE=hard_forecast_value BC_SOFT_TEMPERATURE=1.0
    export FORECAST_VALUE_AUX_COEF=0.10 FORECAST_VALUE_AUX_STRIDE=4 FORECAST_VALUE_AUX_LOOKAHEAD_STEPS=6
    export FORECAST_VALUE_AUX_LOSS=mse FORECAST_VALUE_AUX_TEMPERATURE=1.0 FORECAST_VALUE_RANKING_COEF=0
    export CANDIDATE_INTERACTION_SCORE=0 DIRECT_MASK_ACTION_SCORE=1 DIRECT_MASK_ACTION_PRIMARY=0 FACTORIZED_ACTION_POLICY=0
    export AGENT_CONTEXT_COLUMNS="agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert" CONTEXT_FEATURE_DIM=20
    export TEMPORAL_ENCODER=1 TEMPORAL_HIDDEN_DIM=64 SEPARATE_ACTOR_CRITIC_GRAD_CLIP=1
    export ENT_COEF=0.02 EXCLUDE_SUBTYPE_LATENTS_FROM_STATE=1 NONLINEAR_ACTION_EMBEDDING=1
    export CHANNEL_QUALITY_ENABLED=1 CHANNEL_QUALITY_MODE=condition_dependent_crossover_balanced
    export CHANNEL_QUALITY_SENSOR_IDS="met_station_core radiometer_basic shielded_thermo_hygro surface_temp_ir laser_disdrometer fc4_flux"
    export SENSOR_QUALITY_COLUMNS="agent_context_quality_met_station_core agent_context_quality_radiometer_basic agent_context_quality_shielded_thermo_hygro agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer agent_context_quality_fc4_flux"
    export CHANNEL_QUALITY_MIN_DURATION_STEPS=12 CHANNEL_QUALITY_MAX_DURATION_STEPS=48 CHANNEL_QUALITY_MIN_GAP_STEPS=12
    export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER=6.0 SENSOR_QUALITY_AVAILABILITY_FLOOR=0.1 CHANNEL_QUALITY_DEGRADED_COVERAGE=0.0 CHANNEL_QUALITY_DEGRADED_VALUE=0.1 CHANNEL_QUALITY_REPORT_NOISE_STD=0.02
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v360_terminal_block_credit_cycling_pdppo_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "${i}" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
