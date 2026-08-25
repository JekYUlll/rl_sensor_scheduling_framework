#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-1501 1502 1503 1504 1505}"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2 3 4}"
SCENE_PREFIX=v138_generic_physical_statefix_gate_dev
POLICY_PREFIX=v139_generic_physical_pdppo_dev
RUN_PREFIX=v140_generic_physical_temperature_replay
SENSOR_CFG=configs/sensors/windblown_sensors_flexible_subset_v6_physical_channels.yaml
mkdir -p logs/v140_generic_physical_temperature_replay
pids=()

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
    export RUN_PREFIX CONTROL_SOURCE_RUN_DIR="reports/${SCENE_PREFIX}_seed${seed}_b1p75_20260822"
    export POLICY_SEED="$((seed + 18000))"
    export POLICY_CHECKPOINT_SOURCE="reports/${POLICY_PREFIX}_seed${seed}_b1p75_20260822/custom_ppo.pt"
    export SENSOR_CFG TOTAL_TIMESTEPS=1 TRUTH_STEPS=36000 LOOKBACK=20
    export EXCLUDE_SUBTYPE_LATENTS_FROM_STATE=1
    export EVENT_COVERAGE=0.55 MIN_DURATION=20 MAX_DURATION=64 MIN_GAP=12
    export EVENT_MICROSTRUCTURE_SIGMA=0.12 EVENT_MICROSTRUCTURE_ALPHA=0.15
    export EVENT_PARTICLE_MICROSTRUCTURE_CORRELATION=0.0
    export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY=0.8
    export EVENT_SUBTYPE_ASSIGNMENT=stratified EVENT_SUBTYPE_LATENT_ALPHA=0.15
    export PARTICLE_LATENT_DIAMETER_SCALE=0.28 PARTICLE_LATENT_VELOCITY_SCALE=4.8
    export FLUX_LATENT_SIGMA=2.0 THERMAL_LATENT_SURFACE_SCALE=2.4
    export EVENT_SUBTYPE_CONTEXT_LEAD_STEPS=12 EVENT_SUBTYPE_CONTEXT_NOISE_STD=0.02
    export EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH=1.0
    export ORACLE_EPOCHS=10 ORACLE_FULL_OPEN_REPEAT=3 ORACLE_CANDIDATE_MASK_REPEAT=2
    export ORACLE_SUBTYPE_TEACHER_REPEAT=0 ORACLE_INFERENCE_DEVICE=cuda
    export BUDGET=1.75 STARTUP_BUDGET=2.15 BUDGET_LABEL=b1p75
    export TARGET_WEIGHTS='1 1 1 1 1 1 1 1 1'
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=1 BC_PRETRAIN_LOSS_COEF=0
    export AWBC_COEF=0 SUBTYPE_AUX_COEF=0 SUBTYPE_ACTION_CE_COEF=0
    export SUBTYPE_LOSS_WEIGHTING=0
    export FORECAST_VALUE_AUX_COEF=0
    export CONTEXT_FEATURE_DIM=20 CONTEXT_FUSION_MODE=gated_add
    export TEMPORAL_ENCODER=1 TEMPORAL_HIDDEN_DIM=64
    export TRAINABLE_ACTION_PRIOR=0 NONLINEAR_ACTION_EMBEDDING=1
    export ENT_COEF=0.02 CHANNEL_MARGINAL_ENTROPY_COEF=0
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES=0
    export EVALUATION_POLICY_MODE=deterministic
    export EVALUATION_TEMPERATURE_CANDIDATES='0 0.1 0.25 0.5 0.75 1.0'
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/v140_generic_physical_temperature_replay/seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
