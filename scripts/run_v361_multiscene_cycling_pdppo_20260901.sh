#!/usr/bin/env bash
set -euo pipefail

# V361 tests multi-scene training on the frozen V357 cycling scene family.
# The action geometry, online information, reward, hard constraints, and
# checkpoint rule remain unchanged. Each policy is evaluated on its primary
# scene while training episodes are interleaved across all four scenes. This
# tests scene-distribution robustness without using final-test labels.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=(${SCENE_SEEDS_OVERRIDE:-6901 6902 6903 6904})
POLICY_SEEDS=(${POLICY_SEEDS_OVERRIDE:-7481 7482 7483 7484})
GPU_OFFSET="${GPU_OFFSET:-0}"
RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-v361_multiscene_cycling_pdppo}"
LOG_PREFIX="${LOG_PREFIX_OVERRIDE:-v361_multiscene_cycling_pdppo}"
TRUTH_CSV_ROOT="${TRUTH_CSV_ROOT:-}"
USE_CONTROL_SOURCE="${USE_CONTROL_SOURCE_OVERRIDE:-1}"
DIRECT_MASK_ACTION_PRIMARY="${DIRECT_MASK_ACTION_PRIMARY_OVERRIDE:-0}"
FACTORIZED_ACTION_POLICY="${FACTORIZED_ACTION_POLICY_OVERRIDE:-0}"
DECISION_ONLY_POLICY_UPDATES="${DECISION_ONLY_POLICY_UPDATES_OVERRIDE:-1}"
DECISION_BLOCK_CREDIT="${DECISION_BLOCK_CREDIT_OVERRIDE:-0}"
DECISION_BLOCK_REWARD_MODE="${DECISION_BLOCK_REWARD_MODE_OVERRIDE:-sum}"

run_one() {
  local idx="$1" seed="${SEEDS[$1]}" policy_seed="${POLICY_SEEDS[$1]}"
  local training_sources=""
  for train_seed in "${SEEDS[@]}"; do
    if [[ "$train_seed" != "$seed" ]]; then
      training_sources+=" reports/v357_confirmation_scene_control_seed${train_seed}_b1p75_20260822"
    fi
  done
  (
    export CUDA_VISIBLE_DEVICES="$((idx + GPU_OFFSET))"
    export RUN_PREFIX LOG_PREFIX
    export TOTAL_TIMESTEPS=50000 POLICY_SEED="$policy_seed"
    if [[ "$USE_CONTROL_SOURCE" == "1" ]]; then
      export CONTROL_SOURCE_RUN_DIR="reports/v357_confirmation_scene_control_seed${seed}_b1p75_20260822"
    else
      unset CONTROL_SOURCE_RUN_DIR
    fi
    if [[ -n "$TRUTH_CSV_ROOT" ]]; then
      export TRUTH_CSV="$TRUTH_CSV_ROOT/truth_seed${seed}.csv"
    else
      unset TRUTH_CSV
    fi
    if [[ "$USE_CONTROL_SOURCE" == "1" ]]; then
      export TRAINING_CONTROL_SOURCE_RUN_DIRS="$training_sources"
    else
      unset TRAINING_CONTROL_SOURCE_RUN_DIRS
    fi
    export EVENT_COVERAGE="${EVENT_COVERAGE_OVERRIDE:-0.70}" MIN_DURATION=20 MAX_DURATION=64 MIN_GAP=12 LEAD_STEPS=8
    export EVENT_SUBTYPE_ASSIGNMENT="${EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE:-cycling}" EVENT_SUBTYPE_CYCLE_STEPS="${EVENT_SUBTYPE_CYCLE_STEPS_OVERRIDE:-12}"
    export EVENT_SUBTYPE_PARTICLE_PROB=0.3333333333 EVENT_SUBTYPE_FLUX_PROB=0.3333333333 EVENT_SUBTYPE_THERMAL_PROB=0.3333333334
    export SENSOR_CFG="configs/sensors/windblown_sensors_flexible_subset_v6_physical_channels.yaml"
    export BUDGET="${BUDGET_OVERRIDE:-1.75}" STARTUP_BUDGET="${STARTUP_BUDGET_OVERRIDE:-2.15}" BUDGET_LABEL="${BUDGET_LABEL_OVERRIDE:-b1p75}"
    export REWARD_PROXY_MODE=forecast REWARD_LOSS_NORMALIZATION="${REWARD_LOSS_NORMALIZATION_OVERRIDE:-staticnorm_subtype}" EVENT_START_PROB=1.0
    export GREEDY_LOOKAHEAD_STEPS=6 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=6
    export DECISION_ONLY_POLICY_UPDATES DECISION_BLOCK_CREDIT DECISION_BLOCK_REWARD_MODE
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=5 CHECKPOINT_SELECTION_MIN_UPDATE=5
    export CHECKPOINT_SELECTION_SCORE="${CHECKPOINT_SELECTION_SCORE_OVERRIDE:-oracle_loss_macro_subtype_event_staticnorm}" CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=16384 BC_PRETRAIN_EPOCHS=20 BC_PRETRAIN_LOSS_COEF=1
    export BC_PRETRAIN_DECISION_ONLY=0 BC_PRETRAIN_TARGET_MODE=hard_forecast_value BC_SOFT_TEMPERATURE=1.0
    export FORECAST_VALUE_AUX_COEF=0.10 FORECAST_VALUE_AUX_STRIDE=4 FORECAST_VALUE_AUX_LOOKAHEAD_STEPS=6
    export FORECAST_VALUE_AUX_LOSS="${FORECAST_VALUE_AUX_LOSS_OVERRIDE:-mse}" FORECAST_VALUE_AUX_TEMPERATURE="${FORECAST_VALUE_AUX_TEMPERATURE_OVERRIDE:-1.0}" FORECAST_VALUE_RANKING_COEF="${FORECAST_VALUE_RANKING_COEF_OVERRIDE:-0}"
    export CANDIDATE_INTERACTION_SCORE="${CANDIDATE_INTERACTION_SCORE_OVERRIDE:-0}" CANDIDATE_INTERACTION_SCALE="${CANDIDATE_INTERACTION_SCALE_OVERRIDE:-1.0}" CANDIDATE_INTERACTION_PRIMARY="${CANDIDATE_INTERACTION_PRIMARY_OVERRIDE:-0}" DIRECT_MASK_ACTION_SCORE="${DIRECT_MASK_ACTION_SCORE_OVERRIDE:-1}" DIRECT_MASK_ACTION_PRIMARY FACTORIZED_ACTION_POLICY
    # Nested launchers may deliberately replace subtype alerts with another
    # deployable context family. Preserve that information boundary here.
    export AGENT_CONTEXT_COLUMNS="${AGENT_CONTEXT_COLUMNS_OVERRIDE:-agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert}" CONTEXT_FEATURE_DIM=20
    export TEMPORAL_ENCODER=1 TEMPORAL_HIDDEN_DIM=64 SEPARATE_ACTOR_CRITIC_GRAD_CLIP=1
    export ENT_COEF="${ENT_COEF_OVERRIDE:-0.02}" EXCLUDE_SUBTYPE_LATENTS_FROM_STATE=1 NONLINEAR_ACTION_EMBEDDING=1
    export SUBTYPE_AUX_COEF_OVERRIDE="${SUBTYPE_AUX_COEF_OVERRIDE:-0.3}" SUBTYPE_LOSS_WEIGHTING_OVERRIDE="${SUBTYPE_LOSS_WEIGHTING_OVERRIDE:-1}"
    export CHANNEL_QUALITY_ENABLED=1 CHANNEL_QUALITY_MODE="${CHANNEL_QUALITY_MODE_OVERRIDE:-condition_dependent_crossover_balanced}"
    export CHANNEL_QUALITY_SENSOR_IDS="met_station_core radiometer_basic shielded_thermo_hygro surface_temp_ir laser_disdrometer fc4_flux"
    export SENSOR_QUALITY_COLUMNS="agent_context_quality_met_station_core agent_context_quality_radiometer_basic agent_context_quality_shielded_thermo_hygro agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer agent_context_quality_fc4_flux"
    export CHANNEL_QUALITY_MIN_DURATION_STEPS=12 CHANNEL_QUALITY_MAX_DURATION_STEPS=48 CHANNEL_QUALITY_MIN_GAP_STEPS=12
    export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER=6.0 SENSOR_QUALITY_AVAILABILITY_FLOOR=0.1 CHANNEL_QUALITY_DEGRADED_COVERAGE=0.0 CHANNEL_QUALITY_DEGRADED_VALUE=0.1 CHANNEL_QUALITY_REPORT_NOISE_STD=0.02
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v361_multiscene_cycling_pdppo_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
