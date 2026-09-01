#!/usr/bin/env bash
set -euo pipefail

# V352 is a scene-only structural screen.  It preserves the V338 physical
# channel and cost configuration but changes only the within-event subtype
# temporal structure through the explicit cycling generator mode.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=(6891 6892)
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local slot="$1" seed="$2"
  (
    export CUDA_VISIBLE_DEVICES="$((slot + GPU_OFFSET))"
    export RUN_PREFIX="v352_cycling_scene_control"
    export LOG_PREFIX="v352_cycling_scene_control"
    export TOTAL_TIMESTEPS=1 POLICY_SEED="$((7400 + slot))"
    export EVENT_COVERAGE=0.70 MIN_DURATION=20 MAX_DURATION=64 MIN_GAP=12 LEAD_STEPS=8
    export EVENT_SUBTYPE_ASSIGNMENT=cycling EVENT_SUBTYPE_CYCLE_STEPS=12
    export EVENT_SUBTYPE_PARTICLE_PROB=0.3333333333 EVENT_SUBTYPE_FLUX_PROB=0.3333333333 EVENT_SUBTYPE_THERMAL_PROB=0.3333333334
    export SENSOR_CFG="configs/sensors/windblown_sensors_flexible_subset_v6_physical_channels.yaml"
    export BUDGET=1.75 STARTUP_BUDGET=2.15 BUDGET_LABEL=b1p75
    export VALIDATE_CONTROL_SOURCE_ONLY=0 CONTROL_SOURCE_RUN_DIR=""
    export REWARD_PROXY_MODE=forecast REWARD_LOSS_NORMALIZATION=staticnorm_subtype
    export EVENT_START_PROB=1.0 GREEDY_LOOKAHEAD_STEPS=6 ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS=6
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_RANKING_COEF=0 FORECAST_VALUE_HEAD=0
    export AGENT_CONTEXT_COLUMNS="agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert" CONTEXT_FEATURE_DIM=20
    export EXCLUDE_SUBTYPE_LATENTS_FROM_STATE=1 NONLINEAR_ACTION_EMBEDDING=1
    export CHANNEL_QUALITY_ENABLED=1 CHANNEL_QUALITY_MODE=condition_dependent_crossover_balanced
    export CHANNEL_QUALITY_SENSOR_IDS="met_station_core radiometer_basic shielded_thermo_hygro surface_temp_ir laser_disdrometer fc4_flux"
    export SENSOR_QUALITY_COLUMNS="agent_context_quality_met_station_core agent_context_quality_radiometer_basic agent_context_quality_shielded_thermo_hygro agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer agent_context_quality_fc4_flux"
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v352_cycling_scene_control_seed${seed}.log" 2>&1
}

mkdir -p logs
run_one 0 "${SEEDS[0]}" & p1=$!
run_one 1 "${SEEDS[1]}" & p2=$!
wait "$p1" "$p2"
