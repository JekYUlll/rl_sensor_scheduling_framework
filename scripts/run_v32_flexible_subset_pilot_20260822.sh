#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
DEVICE="${DEVICE:-cuda}"
RUN_PREFIX="${RUN_PREFIX:-v32_flexible_subset_v1_dev}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000}"
TRUTH_STEPS="${TRUTH_STEPS:-36000}"
LOOKBACK="${LOOKBACK:-20}"
EVENT_COVERAGE="${EVENT_COVERAGE:-0.45}"
MIN_DURATION="${MIN_DURATION:-20}"
MAX_DURATION="${MAX_DURATION:-64}"
MIN_GAP="${MIN_GAP:-12}"
LEAD_STEPS="${LEAD_STEPS:-8}"
EVENT_MICROSTRUCTURE_SIGMA="${EVENT_MICROSTRUCTURE_SIGMA:-0.08}"
EVENT_MICROSTRUCTURE_ALPHA="${EVENT_MICROSTRUCTURE_ALPHA:-0.22}"
EVENT_PARTICLE_MICROSTRUCTURE_CORRELATION="${EVENT_PARTICLE_MICROSTRUCTURE_CORRELATION:-0.35}"
SENSOR_CFG="${SENSOR_CFG:-configs/sensors/windblown_sensors_flexible_subset_v1.yaml}"
ORACLE_EPOCHS="${ORACLE_EPOCHS:-10}"
ORACLE_CANDIDATE_MASK_REPEAT="${ORACLE_CANDIDATE_MASK_REPEAT:-1}"
ORACLE_SUBTYPE_TEACHER_REPEAT="${ORACLE_SUBTYPE_TEACHER_REPEAT:-4}"
BUDGET="${BUDGET:-1.35}"
STARTUP_BUDGET="${STARTUP_BUDGET:-1.65}"
BUDGET_LABEL="${BUDGET_LABEL:-b1p35}"
AWBC_COEF="${AWBC_COEF:-0.15}"
AWBC_DECAY_TIMESTEPS="${AWBC_DECAY_TIMESTEPS:-0}"
BC_PRETRAIN_STEPS="${BC_PRETRAIN_STEPS:-1500}"
BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS:-4}"
BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-0.5}"
ENT_COEF="${ENT_COEF:-0.02}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
CHECKPOINT_SELECTION_INTERVAL_UPDATES="${CHECKPOINT_SELECTION_INTERVAL_UPDATES:-0}"
TRAINABLE_ACTION_PRIOR="${TRAINABLE_ACTION_PRIOR:-1}"
NONLINEAR_ACTION_EMBEDDING="${NONLINEAR_ACTION_EMBEDDING:-0}"
EVENT_SUBTYPE_LATENT_ALPHA="${EVENT_SUBTYPE_LATENT_ALPHA:-0.22}"
PARTICLE_HUMIDITY_BOOST="${PARTICLE_HUMIDITY_BOOST:-1.0}"
FLUX_WIND_BOOST="${FLUX_WIND_BOOST:-1.0}"
THERMAL_AIR_TEMP_DROP="${THERMAL_AIR_TEMP_DROP:-1.0}"
PARTICLE_LATENT_DIAMETER_SCALE="${PARTICLE_LATENT_DIAMETER_SCALE:-0.14}"
PARTICLE_LATENT_VELOCITY_SCALE="${PARTICLE_LATENT_VELOCITY_SCALE:-2.4}"
FLUX_LATENT_SIGMA="${FLUX_LATENT_SIGMA:-1.2}"
THERMAL_LATENT_SURFACE_SCALE="${THERMAL_LATENT_SURFACE_SCALE:-2.4}"
EVENT_SUBTYPE_ASSIGNMENT="${EVENT_SUBTYPE_ASSIGNMENT:-random}"
read -r -a TEACHER_CALM_SENSOR_ARGS <<< "${TEACHER_CALM_SENSORS:-met_station_core radiometer_basic}"
read -r -a TEACHER_PARTICLE_SENSOR_ARGS <<< "${TEACHER_PARTICLE_SENSORS:-met_station_core laser_disdrometer}"
read -r -a TEACHER_FLUX_SENSOR_ARGS <<< "${TEACHER_FLUX_SENSORS:-met_station_core fc4_flux}"
read -r -a TEACHER_THERMAL_SENSOR_ARGS <<< "${TEACHER_THERMAL_SENSORS:-shielded_thermo_hygro surface_temp_ir}"
REWARD_LOSS_NORMALIZATION="${REWARD_LOSS_NORMALIZATION:-none}"
AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_static_auto}"
SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-0.3}"
SUBTYPE_ACTION_CE_COEF="${SUBTYPE_ACTION_CE_COEF:-0.0}"
SUBTYPE_ACTION_EVENT_ONLY="${SUBTYPE_ACTION_EVENT_ONLY:-0}"
SEPARATE_ACTOR_CRITIC_GRAD_CLIP="${SEPARATE_ACTOR_CRITIC_GRAD_CLIP:-1}"
CONTROL_SOURCE_RUN_DIR="${CONTROL_SOURCE_RUN_DIR:-}"
VALIDATE_CONTROL_SOURCE_ONLY="${VALIDATE_CONTROL_SOURCE_ONLY:-0}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(401)
fi

for seed in "${SEEDS[@]}"; do
  out_dir="reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_20260822"
  mkdir -p "$out_dir"
  "$PY" scripts/97_v32_flexible_subset_preflight.py \
    --sensor-cfg "$SENSOR_CFG" \
    --budget "$BUDGET" \
    --startup-peak-budget "$STARTUP_BUDGET" \
    --output "${out_dir}/action_geometry.json" \
    > "${out_dir}/action_geometry.stdout.json"

  control_args=()
  if [[ -n "$CONTROL_SOURCE_RUN_DIR" ]]; then
    control_args+=(--control-source-run-dir "$CONTROL_SOURCE_RUN_DIR")
  fi
  if [[ "$VALIDATE_CONTROL_SOURCE_ONLY" == "1" ]]; then
    control_args+=(--validate-control-source-only)
  fi
  if [[ "$SEPARATE_ACTOR_CRITIC_GRAD_CLIP" == "1" ]]; then
    control_args+=(--separate-actor-critic-grad-clip)
  else
    control_args+=(--no-separate-actor-critic-grad-clip)
  fi
  if [[ "$TRAINABLE_ACTION_PRIOR" == "1" ]]; then
    control_args+=(--trainable-action-prior)
  else
    control_args+=(--no-trainable-action-prior)
  fi
  if [[ "$NONLINEAR_ACTION_EMBEDDING" == "1" ]]; then
    control_args+=(--nonlinear-action-embedding)
  else
    control_args+=(--no-nonlinear-action-embedding)
  fi
  if [[ "$SUBTYPE_ACTION_EVENT_ONLY" == "1" ]]; then
    control_args+=(--subtype-action-event-only)
  else
    control_args+=(--no-subtype-action-event-only)
  fi

  "$PY" scripts/58_v31_split_protocol_run.py \
    --out-dir "$out_dir" \
    "${control_args[@]}" \
    --sensor-cfg "$SENSOR_CFG" \
    --seed "$seed" \
    --budget "$BUDGET" \
    --startup-peak-budget "$STARTUP_BUDGET" \
    --truth-steps "$TRUTH_STEPS" \
    --freq-s 3600 \
    --lookback "$LOOKBACK" \
    --split-ratios 0.35 0.50 0.075 0.075 \
    --event-coverage "$EVENT_COVERAGE" \
    --min-duration "$MIN_DURATION" \
    --max-duration "$MAX_DURATION" \
    --min-gap "$MIN_GAP" \
    --lead-steps "$LEAD_STEPS" \
    --wind-margin-ms 1.4 \
    --cred-hysteresis-on 0.6 \
    --cred-hysteresis-off 0.3 \
    --flux-wind-exponent 3.0 \
    --event-microstructure-sigma "$EVENT_MICROSTRUCTURE_SIGMA" \
    --event-microstructure-alpha "$EVENT_MICROSTRUCTURE_ALPHA" \
    --event-microstructure-diameter-scale 0.08 \
    --event-microstructure-velocity-scale 0.20 \
    --event-particle-microstructure-correlation "$EVENT_PARTICLE_MICROSTRUCTURE_CORRELATION" \
    --event-subtypes-enabled \
    --event-subtype-assignment "$EVENT_SUBTYPE_ASSIGNMENT" \
    --event-subtype-particle-prob 0.36 \
    --event-subtype-flux-prob 0.36 \
    --event-subtype-thermal-prob 0.28 \
    --event-subtype-particle-flux-multiplier 0.75 \
    --event-subtype-flux-multiplier 3.5 \
    --event-subtype-thermal-flux-multiplier 0.65 \
    --event-subtype-particle-diameter-shift-mm 0.12 \
    --event-subtype-particle-velocity-boost-ms 1.8 \
    --event-subtype-flux-diameter-shift-mm -0.05 \
    --event-subtype-flux-velocity-boost-ms 0.8 \
    --event-subtype-thermal-surface-drop-c 2.4 \
    --event-subtype-particle-humidity-boost-pct "$PARTICLE_HUMIDITY_BOOST" \
    --event-subtype-flux-wind-boost-ms "$FLUX_WIND_BOOST" \
    --event-subtype-thermal-air-temp-drop-c "$THERMAL_AIR_TEMP_DROP" \
    --event-subtype-latent-alpha "$EVENT_SUBTYPE_LATENT_ALPHA" \
    --event-subtype-particle-latent-diameter-scale-mm "$PARTICLE_LATENT_DIAMETER_SCALE" \
    --event-subtype-particle-latent-velocity-scale-ms "$PARTICLE_LATENT_VELOCITY_SCALE" \
    --event-subtype-flux-latent-sigma "$FLUX_LATENT_SIGMA" \
    --event-subtype-thermal-latent-surface-scale-c "$THERMAL_LATENT_SURFACE_SCALE" \
    --event-subtype-latent-target-lag-steps 4 \
    --event-subtype-context-lead-steps 8 \
    --event-subtype-context-noise-std 0.05 \
    --oracle-rollout-steps 2048 \
    --oracle-rollouts-per-policy 4 \
    --oracle-epochs "$ORACLE_EPOCHS" \
    --oracle-batch-size 512 \
    --oracle-loss-clip 20 \
    --oracle-candidate-mask-repeat "$ORACLE_CANDIDATE_MASK_REPEAT" \
    --oracle-candidate-mask-limit 0 \
    --oracle-subtype-teacher-repeat "$ORACLE_SUBTYPE_TEACHER_REPEAT" \
    --oracle-subtype-teacher-lookahead-steps 8 \
    --oracle-subtype-teacher-calm-sensors "${TEACHER_CALM_SENSOR_ARGS[@]}" \
    --oracle-subtype-teacher-particle-sensors "${TEACHER_PARTICLE_SENSOR_ARGS[@]}" \
    --oracle-subtype-teacher-flux-sensors "${TEACHER_FLUX_SENSOR_ARGS[@]}" \
    --oracle-subtype-teacher-thermal-sensors "${TEACHER_THERMAL_SENSOR_ARGS[@]}" \
    --oracle-device auto \
    --oracle-inference-device cpu \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-steps 1024 \
    --batch-size 128 \
    --n-epochs 8 \
    --learning-rate "$LEARNING_RATE" \
    --ent-coef "$ENT_COEF" \
    --awbc-coef "$AWBC_COEF" \
    --awbc-decay-timesteps "$AWBC_DECAY_TIMESTEPS" \
    --awbc-label-stride 4 \
    --checkpoint-selection-interval-updates "$CHECKPOINT_SELECTION_INTERVAL_UPDATES" \
    --bc-pretrain-steps "$BC_PRETRAIN_STEPS" \
    --bc-pretrain-epochs "$BC_PRETRAIN_EPOCHS" \
    --bc-pretrain-batch-size 256 \
    --bc-pretrain-loss-coef "$BC_PRETRAIN_LOSS_COEF" \
    --subtype-aux-coef "$SUBTYPE_AUX_COEF" \
    --subtype-aux-classes 4 \
    --subtype-aux-lookahead-steps 8 \
    --subtype-action-ce-coef "$SUBTYPE_ACTION_CE_COEF" \
    --no-subtype-router \
    --awbc-teacher-mode "$AWBC_TEACHER_MODE" \
    --awbc-teacher-subtype-calm-sensors "${TEACHER_CALM_SENSOR_ARGS[@]}" \
    --awbc-teacher-subtype-particle-sensors "${TEACHER_PARTICLE_SENSOR_ARGS[@]}" \
    --awbc-teacher-subtype-flux-sensors "${TEACHER_FLUX_SENSOR_ARGS[@]}" \
    --awbc-teacher-subtype-thermal-sensors "${TEACHER_THERMAL_SENSOR_ARGS[@]}" \
    --awbc-teacher-auto-score-mode staticnorm \
    --awbc-teacher-event-lookahead-steps 8 \
    --awbc-teacher-dwell-steps 6 \
    --prior-kl-coef 0.0 \
    --greedy-lookahead-steps 4 \
    --event-start-prob 0.70 \
    --event-aware-critic \
    --no-event-gated-actor \
    --context-encoder \
    --context-feature-dim 10 \
    --context-hidden-dim 64 \
    --context-fusion-mode gated_add \
    --context-layer-norm \
    --include-alert-context-features \
    --no-include-event-flag-in-state \
    --soc-aux-horizon 0 \
    --soc-aux-coef 0.0 \
    --train-episode-len 512 \
    --no-use-candidate-prior \
    --static-selection-steps 384 \
    --static-selection-rollouts 4 \
    --eval-steps 384 \
    --eval-rollouts 6 \
    --eval-start-selection subtype_balanced_transport_rich \
    --eval-event-fraction 0.70 \
    --eval-selection-stride 48 \
    --lambda-warmup-abort 1.0 \
    --lambda-switch 0.002 \
    --event-reward-multiplier 1.0 \
    --reward-loss-normalization "$REWARD_LOSS_NORMALIZATION" \
    --lambda-duty-balance 0.0 \
    --duty-score-feedback 0.0 \
    --no-duty-hard-guard \
    --no-primary-eval-duty-guard \
    --min-dwell-steps 6 \
    --target-weights 0.25 0.35 0.30 0.10 0.10 0.20 18.0 8.0 8.0 \
    --subtype-loss-weighting \
    --subtype-particle-target-weights 0.10 0.10 0.20 0.05 0.05 0.10 4.0 14.0 14.0 \
    --subtype-flux-target-weights 0.10 0.10 0.30 0.05 0.05 0.10 24.0 4.0 4.0 \
    --subtype-thermal-target-weights 0.50 10.0 0.20 0.05 0.05 0.20 2.0 2.0 2.0 \
    --disable-coverage-groups \
    --device "$DEVICE" \
    2>&1 | tee "${out_dir}/run_train_eval.log"
done
