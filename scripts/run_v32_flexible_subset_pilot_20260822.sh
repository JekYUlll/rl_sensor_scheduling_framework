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
EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY="${EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY:-0}"
SENSOR_CFG="${SENSOR_CFG:-configs/sensors/windblown_sensors_flexible_subset_v1.yaml}"
ORACLE_EPOCHS="${ORACLE_EPOCHS:-10}"
ORACLE_TYPE="${ORACLE_TYPE:-tcn}"
ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
ORACLE_FULL_OPEN_REPEAT="${ORACLE_FULL_OPEN_REPEAT:-3}"
ORACLE_CANDIDATE_MASK_REPEAT="${ORACLE_CANDIDATE_MASK_REPEAT:-1}"
ORACLE_SUBTYPE_TEACHER_REPEAT="${ORACLE_SUBTYPE_TEACHER_REPEAT:-4}"
BUDGET="${BUDGET:-1.35}"
STARTUP_BUDGET="${STARTUP_BUDGET:-1.65}"
BUDGET_LABEL="${BUDGET_LABEL:-b1p35}"
AWBC_COEF="${AWBC_COEF:-0.15}"
AWBC_DECAY_TIMESTEPS="${AWBC_DECAY_TIMESTEPS:-0}"
AWBC_EVENT_ONLY="${AWBC_EVENT_ONLY:-0}"
AWBC_LABEL_STRIDE="${AWBC_LABEL_STRIDE:-4}"
BC_PRETRAIN_STEPS="${BC_PRETRAIN_STEPS:-1500}"
BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS:-4}"
BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-0.5}"
BC_PRETRAIN_TARGET_MODE="${BC_PRETRAIN_TARGET_MODE:-hard}"
BC_SOFT_TEMPERATURE="${BC_SOFT_TEMPERATURE:-1.0}"
FORECAST_VALUE_AUX_COEF="${FORECAST_VALUE_AUX_COEF:-0.0}"
FORECAST_VALUE_AUX_STRIDE="${FORECAST_VALUE_AUX_STRIDE:-64}"
FORECAST_VALUE_AUX_LOOKAHEAD_STEPS="${FORECAST_VALUE_AUX_LOOKAHEAD_STEPS:-0}"
FORECAST_VALUE_AUX_LOSS="${FORECAST_VALUE_AUX_LOSS:-mse}"
FORECAST_VALUE_AUX_TEMPERATURE="${FORECAST_VALUE_AUX_TEMPERATURE:-1.0}"
FORECAST_VALUE_HEAD="${FORECAST_VALUE_HEAD:-0}"
FORECAST_VALUE_HEAD_SCALE="${FORECAST_VALUE_HEAD_SCALE:-1.0}"
FORECAST_VALUE_HEAD_HIDDEN_DIM="${FORECAST_VALUE_HEAD_HIDDEN_DIM:-128}"
FORECAST_VALUE_HEAD_MODE="${FORECAST_VALUE_HEAD_MODE:-factorized}"
ENT_COEF="${ENT_COEF:-0.02}"
CHANNEL_MARGINAL_ENTROPY_COEF="${CHANNEL_MARGINAL_ENTROPY_COEF:-0}"
POLICY_INIT_SOURCE="${POLICY_INIT_SOURCE:-}"
POLICY_CHECKPOINT_SOURCE="${POLICY_CHECKPOINT_SOURCE:-}"
TRAINING_CONTROL_SOURCE_RUN_DIRS="${TRAINING_CONTROL_SOURCE_RUN_DIRS:-}"
EVALUATION_POLICY_MODE="${EVALUATION_POLICY_MODE:-deterministic}"
EVALUATION_SAMPLING_SEED="${EVALUATION_SAMPLING_SEED:-}"
EVALUATION_SAMPLING_TEMPERATURE="${EVALUATION_SAMPLING_TEMPERATURE:-1.0}"
EVALUATION_TEMPERATURE_CANDIDATES="${EVALUATION_TEMPERATURE_CANDIDATES:-}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
GREEDY_LOOKAHEAD_STEPS="${GREEDY_LOOKAHEAD_STEPS:-4}"
CHECKPOINT_SELECTION_INTERVAL_UPDATES="${CHECKPOINT_SELECTION_INTERVAL_UPDATES:-0}"
CHECKPOINT_SELECTION_SCORE="${CHECKPOINT_SELECTION_SCORE:-oracle_loss_mean}"
CHECKPOINT_REQUIRE_VALID_BEHAVIOR="${CHECKPOINT_REQUIRE_VALID_BEHAVIOR:-0}"
TRAINABLE_ACTION_PRIOR="${TRAINABLE_ACTION_PRIOR:-1}"
NONLINEAR_ACTION_EMBEDDING="${NONLINEAR_ACTION_EMBEDDING:-0}"
EVENT_SUBTYPE_LATENT_ALPHA="${EVENT_SUBTYPE_LATENT_ALPHA:-0.22}"
EVENT_SUBTYPE_TARGET_LAG_STEPS="${EVENT_SUBTYPE_TARGET_LAG_STEPS:-4}"
EVENT_SUBTYPE_CONTEXT_LEAD_STEPS="${EVENT_SUBTYPE_CONTEXT_LEAD_STEPS:-8}"
EVENT_SUBTYPE_CONTEXT_NOISE_STD="${EVENT_SUBTYPE_CONTEXT_NOISE_STD:-0.05}"
EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH="${EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH:-0.0}"
PARTICLE_HUMIDITY_BOOST="${PARTICLE_HUMIDITY_BOOST:-1.0}"
FLUX_WIND_BOOST="${FLUX_WIND_BOOST:-1.0}"
THERMAL_AIR_TEMP_DROP="${THERMAL_AIR_TEMP_DROP:-1.0}"
PARTICLE_LATENT_DIAMETER_SCALE="${PARTICLE_LATENT_DIAMETER_SCALE:-0.14}"
PARTICLE_LATENT_VELOCITY_SCALE="${PARTICLE_LATENT_VELOCITY_SCALE:-2.4}"
FLUX_LATENT_SIGMA="${FLUX_LATENT_SIGMA:-1.2}"
THERMAL_LATENT_SURFACE_SCALE="${THERMAL_LATENT_SURFACE_SCALE:-2.4}"
EVENT_SUBTYPE_ASSIGNMENT="${EVENT_SUBTYPE_ASSIGNMENT:-random}"
read -r -a TARGET_WEIGHT_ARGS <<< "${TARGET_WEIGHTS:-0.25 0.35 0.30 0.10 0.10 0.20 18.0 8.0 8.0}"
read -r -a PARTICLE_TARGET_WEIGHT_ARGS <<< "${PARTICLE_TARGET_WEIGHTS:-0.10 0.10 0.20 0.05 0.05 0.10 4.0 14.0 14.0}"
read -r -a FLUX_TARGET_WEIGHT_ARGS <<< "${FLUX_TARGET_WEIGHTS:-0.10 0.10 0.30 0.05 0.05 0.10 24.0 4.0 4.0}"
read -r -a THERMAL_TARGET_WEIGHT_ARGS <<< "${THERMAL_TARGET_WEIGHTS:-0.50 10.0 0.20 0.05 0.05 0.20 2.0 2.0 2.0}"
read -r -a TEACHER_CALM_SENSOR_ARGS <<< "${TEACHER_CALM_SENSORS:-met_station_core radiometer_basic}"
read -r -a TEACHER_PARTICLE_SENSOR_ARGS <<< "${TEACHER_PARTICLE_SENSORS:-met_station_core laser_disdrometer}"
read -r -a TEACHER_FLUX_SENSOR_ARGS <<< "${TEACHER_FLUX_SENSORS:-met_station_core fc4_flux}"
read -r -a TEACHER_THERMAL_SENSOR_ARGS <<< "${TEACHER_THERMAL_SENSORS:-shielded_thermo_hygro surface_temp_ir}"
read -r -a AWBC_TEACHER_CALM_SENSOR_ARGS <<< "${AWBC_TEACHER_CALM_SENSORS:-${TEACHER_CALM_SENSORS:-met_station_core radiometer_basic}}"
read -r -a AWBC_TEACHER_PARTICLE_SENSOR_ARGS <<< "${AWBC_TEACHER_PARTICLE_SENSORS:-${TEACHER_PARTICLE_SENSORS:-met_station_core laser_disdrometer}}"
read -r -a AWBC_TEACHER_FLUX_SENSOR_ARGS <<< "${AWBC_TEACHER_FLUX_SENSORS:-${TEACHER_FLUX_SENSORS:-met_station_core fc4_flux}}"
read -r -a AWBC_TEACHER_THERMAL_SENSOR_ARGS <<< "${AWBC_TEACHER_THERMAL_SENSORS:-${TEACHER_THERMAL_SENSORS:-shielded_thermo_hygro surface_temp_ir}}"
REWARD_LOSS_NORMALIZATION="${REWARD_LOSS_NORMALIZATION:-none}"
REWARD_PROXY_MODE="${REWARD_PROXY_MODE:-forecast}"
AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_static_auto}"
AWBC_TEACHER_ALERT_THRESHOLD="${AWBC_TEACHER_ALERT_THRESHOLD:-0.5}"
SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-0.3}"
SUBTYPE_ACTION_CE_COEF="${SUBTYPE_ACTION_CE_COEF:-0.0}"
SUBTYPE_ACTION_SUPERVISION_MODE="${SUBTYPE_ACTION_SUPERVISION_MODE:-exact_action}"
SUBTYPE_ACTION_EVENT_ONLY="${SUBTYPE_ACTION_EVENT_ONLY:-0}"
SUBTYPE_LOSS_WEIGHTING="${SUBTYPE_LOSS_WEIGHTING:-1}"
CONTEXT_FEATURE_DIM="${CONTEXT_FEATURE_DIM:-10}"
CONTEXT_FUSION_MODE="${CONTEXT_FUSION_MODE:-gated_add}"
ALIGNED_QUALITY_ACTION_SCORE="${ALIGNED_QUALITY_ACTION_SCORE:-0}"
TEMPORAL_ENCODER="${TEMPORAL_ENCODER:-0}"
TEMPORAL_HIDDEN_DIM="${TEMPORAL_HIDDEN_DIM:-64}"
MEASUREMENT_UPDATE_MODE="${MEASUREMENT_UPDATE_MODE:-direct}"
COMMON_RANDOM_NUMBERS="${COMMON_RANDOM_NUMBERS:-0}"
CHANNEL_QUALITY_ENABLED="${CHANNEL_QUALITY_ENABLED:-0}"
CHANNEL_QUALITY_DEGRADED_COVERAGE="${CHANNEL_QUALITY_DEGRADED_COVERAGE:-0.0}"
CHANNEL_QUALITY_MIN_DURATION_STEPS="${CHANNEL_QUALITY_MIN_DURATION_STEPS:-12}"
CHANNEL_QUALITY_MAX_DURATION_STEPS="${CHANNEL_QUALITY_MAX_DURATION_STEPS:-48}"
CHANNEL_QUALITY_MIN_GAP_STEPS="${CHANNEL_QUALITY_MIN_GAP_STEPS:-12}"
CHANNEL_QUALITY_DEGRADED_VALUE="${CHANNEL_QUALITY_DEGRADED_VALUE:-0.2}"
CHANNEL_QUALITY_TRANSITION_STEPS="${CHANNEL_QUALITY_TRANSITION_STEPS:-0}"
CHANNEL_QUALITY_REPORT_NOISE_STD="${CHANNEL_QUALITY_REPORT_NOISE_STD:-0.02}"
SENSOR_QUALITY_MAX_NOISE_MULTIPLIER="${SENSOR_QUALITY_MAX_NOISE_MULTIPLIER:-1.0}"
SENSOR_QUALITY_AVAILABILITY_FLOOR="${SENSOR_QUALITY_AVAILABILITY_FLOOR:-1.0}"
read -r -a CHANNEL_QUALITY_SENSOR_ARGS <<< "${CHANNEL_QUALITY_SENSOR_IDS:-met_station_core radiometer_basic shielded_thermo_hygro surface_temp_ir laser_disdrometer fc4_flux}"
read -r -a SENSOR_QUALITY_COLUMN_ARGS <<< "${SENSOR_QUALITY_COLUMNS:-agent_context_quality_met_station_core agent_context_quality_radiometer_basic agent_context_quality_shielded_thermo_hygro agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer agent_context_quality_fc4_flux}"
EXCLUDE_SUBTYPE_LATENTS_FROM_STATE="${EXCLUDE_SUBTYPE_LATENTS_FROM_STATE:-0}"
SEPARATE_ACTOR_CRITIC_GRAD_CLIP="${SEPARATE_ACTOR_CRITIC_GRAD_CLIP:-1}"
CONTROL_SOURCE_RUN_DIR="${CONTROL_SOURCE_RUN_DIR:-}"
VALIDATE_CONTROL_SOURCE_ONLY="${VALIDATE_CONTROL_SOURCE_ONLY:-0}"
POLICY_SEED="${POLICY_SEED:-}"

TEMPORAL_ARGS=(--no-temporal-encoder)
if [[ "$TEMPORAL_ENCODER" == "1" ]]; then
  TEMPORAL_ARGS=(--temporal-encoder)
fi
QUALITY_ARGS=(--no-channel-quality-enabled)
if [[ "$CHANNEL_QUALITY_ENABLED" == "1" ]]; then
  QUALITY_ARGS=(
    --channel-quality-enabled
    --channel-quality-sensor-ids "${CHANNEL_QUALITY_SENSOR_ARGS[@]}"
    --channel-quality-degraded-coverage "$CHANNEL_QUALITY_DEGRADED_COVERAGE"
    --channel-quality-min-duration-steps "$CHANNEL_QUALITY_MIN_DURATION_STEPS"
    --channel-quality-max-duration-steps "$CHANNEL_QUALITY_MAX_DURATION_STEPS"
    --channel-quality-min-gap-steps "$CHANNEL_QUALITY_MIN_GAP_STEPS"
    --channel-quality-degraded-value "$CHANNEL_QUALITY_DEGRADED_VALUE"
    --channel-quality-transition-steps "$CHANNEL_QUALITY_TRANSITION_STEPS"
    --channel-quality-report-noise-std "$CHANNEL_QUALITY_REPORT_NOISE_STD"
    --sensor-quality-columns "${SENSOR_QUALITY_COLUMN_ARGS[@]}"
    --sensor-quality-max-noise-multiplier "$SENSOR_QUALITY_MAX_NOISE_MULTIPLIER"
    --sensor-quality-availability-floor "$SENSOR_QUALITY_AVAILABILITY_FLOOR"
  )
fi
CRN_ARGS=(--no-common-random-numbers)
if [[ "$COMMON_RANDOM_NUMBERS" == "1" ]]; then
  CRN_ARGS=(--common-random-numbers)
fi
QUALITY_SCORE_ARGS=(--no-aligned-quality-action-score)
if [[ "$ALIGNED_QUALITY_ACTION_SCORE" == "1" ]]; then
  QUALITY_SCORE_ARGS=(--aligned-quality-action-score)
fi

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
  if [[ -n "$POLICY_SEED" ]]; then
    control_args+=(--policy-seed "$POLICY_SEED")
  fi
  if [[ -n "$POLICY_INIT_SOURCE" ]]; then
    control_args+=(--policy-init-source "$POLICY_INIT_SOURCE")
  fi
  if [[ -n "$POLICY_CHECKPOINT_SOURCE" ]]; then
    control_args+=(--policy-checkpoint-source "$POLICY_CHECKPOINT_SOURCE")
  fi
  if [[ -n "$TRAINING_CONTROL_SOURCE_RUN_DIRS" ]]; then
    read -r -a training_control_sources <<< "$TRAINING_CONTROL_SOURCE_RUN_DIRS"
    control_args+=(--training-control-source-run-dirs "${training_control_sources[@]}")
  fi
  control_args+=(--evaluation-policy-mode "$EVALUATION_POLICY_MODE")
  if [[ -n "$EVALUATION_SAMPLING_SEED" ]]; then
    control_args+=(--evaluation-sampling-seed "$EVALUATION_SAMPLING_SEED")
  fi
  control_args+=(--evaluation-sampling-temperature "$EVALUATION_SAMPLING_TEMPERATURE")
  if [[ "$FORECAST_VALUE_HEAD" == "1" ]]; then
    control_args+=(--forecast-value-head)
  else
    control_args+=(--no-forecast-value-head)
  fi
  control_args+=(--forecast-value-head-scale "$FORECAST_VALUE_HEAD_SCALE")
  control_args+=(--forecast-value-head-hidden-dim "$FORECAST_VALUE_HEAD_HIDDEN_DIM")
  control_args+=(--forecast-value-head-mode "$FORECAST_VALUE_HEAD_MODE")
  if [[ -n "$EVALUATION_TEMPERATURE_CANDIDATES" ]]; then
    read -r -a evaluation_temperature_args <<< "$EVALUATION_TEMPERATURE_CANDIDATES"
    control_args+=(--evaluation-temperature-candidates "${evaluation_temperature_args[@]}")
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
  if [[ "$AWBC_EVENT_ONLY" == "1" ]]; then
    control_args+=(--awbc-event-only)
  else
    control_args+=(--no-awbc-event-only)
  fi
  if [[ "$SUBTYPE_LOSS_WEIGHTING" == "1" ]]; then
    control_args+=(--subtype-loss-weighting)
  else
    control_args+=(--no-subtype-loss-weighting)
  fi
  if [[ "$EXCLUDE_SUBTYPE_LATENTS_FROM_STATE" == "1" ]]; then
    control_args+=(--exclude-subtype-latents-from-state)
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
    --event-subtype-particle-min-parsivel-availability "$EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY" \
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
    --event-subtype-latent-target-lag-steps "$EVENT_SUBTYPE_TARGET_LAG_STEPS" \
    --event-subtype-context-lead-steps "$EVENT_SUBTYPE_CONTEXT_LEAD_STEPS" \
    --event-subtype-context-noise-std "$EVENT_SUBTYPE_CONTEXT_NOISE_STD" \
    --event-subtype-context-latent-strength "$EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH" \
    "${QUALITY_ARGS[@]}" \
    --oracle-rollout-steps 2048 \
    --oracle-type "$ORACLE_TYPE" \
    --oracle-rollouts-per-policy 4 \
    --oracle-epochs "$ORACLE_EPOCHS" \
    --oracle-full-open-repeat "$ORACLE_FULL_OPEN_REPEAT" \
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
    --oracle-inference-device "$ORACLE_INFERENCE_DEVICE" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-steps 1024 \
    --batch-size 128 \
    --n-epochs 8 \
    --learning-rate "$LEARNING_RATE" \
    --ent-coef "$ENT_COEF" \
    --channel-marginal-entropy-coef "$CHANNEL_MARGINAL_ENTROPY_COEF" \
    --awbc-coef "$AWBC_COEF" \
    --awbc-decay-timesteps "$AWBC_DECAY_TIMESTEPS" \
    --awbc-label-stride "$AWBC_LABEL_STRIDE" \
    --checkpoint-selection-interval-updates "$CHECKPOINT_SELECTION_INTERVAL_UPDATES" \
    "$([[ "$CHECKPOINT_REQUIRE_VALID_BEHAVIOR" == "1" ]] && printf '%s' --checkpoint-require-valid-behavior || printf '%s' --no-checkpoint-require-valid-behavior)" \
    --checkpoint-selection-score "$CHECKPOINT_SELECTION_SCORE" \
    --bc-pretrain-steps "$BC_PRETRAIN_STEPS" \
    --bc-pretrain-epochs "$BC_PRETRAIN_EPOCHS" \
    --bc-pretrain-batch-size 256 \
    --bc-pretrain-loss-coef "$BC_PRETRAIN_LOSS_COEF" \
    --bc-pretrain-target-mode "$BC_PRETRAIN_TARGET_MODE" \
    --bc-soft-temperature "$BC_SOFT_TEMPERATURE" \
    --forecast-value-aux-coef "$FORECAST_VALUE_AUX_COEF" \
    --forecast-value-aux-stride "$FORECAST_VALUE_AUX_STRIDE" \
    --forecast-value-aux-lookahead-steps "$FORECAST_VALUE_AUX_LOOKAHEAD_STEPS" \
    --forecast-value-aux-loss "$FORECAST_VALUE_AUX_LOSS" \
    --forecast-value-aux-temperature "$FORECAST_VALUE_AUX_TEMPERATURE" \
    --subtype-aux-coef "$SUBTYPE_AUX_COEF" \
    --subtype-aux-classes 4 \
    --subtype-aux-lookahead-steps 8 \
    --subtype-action-ce-coef "$SUBTYPE_ACTION_CE_COEF" \
    --subtype-action-supervision-mode "$SUBTYPE_ACTION_SUPERVISION_MODE" \
    --no-subtype-router \
    --awbc-teacher-mode "$AWBC_TEACHER_MODE" \
    --awbc-teacher-subtype-calm-sensors "${AWBC_TEACHER_CALM_SENSOR_ARGS[@]}" \
    --awbc-teacher-subtype-particle-sensors "${AWBC_TEACHER_PARTICLE_SENSOR_ARGS[@]}" \
    --awbc-teacher-subtype-flux-sensors "${AWBC_TEACHER_FLUX_SENSOR_ARGS[@]}" \
    --awbc-teacher-subtype-thermal-sensors "${AWBC_TEACHER_THERMAL_SENSOR_ARGS[@]}" \
    --awbc-teacher-auto-score-mode staticnorm \
    --awbc-teacher-event-lookahead-steps 8 \
    --awbc-teacher-alert-threshold "$AWBC_TEACHER_ALERT_THRESHOLD" \
    --awbc-teacher-dwell-steps 6 \
    --prior-kl-coef 0.0 \
    --greedy-lookahead-steps "$GREEDY_LOOKAHEAD_STEPS" \
    --event-start-prob 0.70 \
    --event-aware-critic \
    --no-event-gated-actor \
    --context-encoder \
    --context-feature-dim "$CONTEXT_FEATURE_DIM" \
    --measurement-update-mode "$MEASUREMENT_UPDATE_MODE" \
    "${CRN_ARGS[@]}" \
    --context-hidden-dim 64 \
    --context-fusion-mode "$CONTEXT_FUSION_MODE" \
    --context-layer-norm \
    "${QUALITY_SCORE_ARGS[@]}" \
    "${TEMPORAL_ARGS[@]}" \
    --temporal-hidden-dim "$TEMPORAL_HIDDEN_DIM" \
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
    --reward-proxy-mode "$REWARD_PROXY_MODE" \
    --lambda-duty-balance 0.0 \
    --duty-score-feedback 0.0 \
    --no-duty-hard-guard \
    --no-primary-eval-duty-guard \
    --min-dwell-steps 6 \
    --target-weights "${TARGET_WEIGHT_ARGS[@]}" \
    --subtype-particle-target-weights "${PARTICLE_TARGET_WEIGHT_ARGS[@]}" \
    --subtype-flux-target-weights "${FLUX_TARGET_WEIGHT_ARGS[@]}" \
    --subtype-thermal-target-weights "${THERMAL_TARGET_WEIGHT_ARGS[@]}" \
    --disable-coverage-groups \
    --device "$DEVICE" \
    2>&1 | tee "${out_dir}/run_train_eval.log"
done
