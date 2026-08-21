#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
DEVICE="${DEVICE:-cuda}"
RUN_PREFIX="${RUN_PREFIX:-v32_flexible_subset_v1_dev}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000}"
TRUTH_STEPS="${TRUTH_STEPS:-36000}"
SENSOR_CFG="${SENSOR_CFG:-configs/sensors/windblown_sensors_flexible_subset_v1.yaml}"
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
REWARD_LOSS_NORMALIZATION="${REWARD_LOSS_NORMALIZATION:-none}"
AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_static_auto}"
SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-0.3}"
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

  "$PY" scripts/58_v31_split_protocol_run.py \
    --out-dir "$out_dir" \
    "${control_args[@]}" \
    --sensor-cfg "$SENSOR_CFG" \
    --seed "$seed" \
    --budget "$BUDGET" \
    --startup-peak-budget "$STARTUP_BUDGET" \
    --truth-steps "$TRUTH_STEPS" \
    --freq-s 3600 \
    --split-ratios 0.35 0.50 0.075 0.075 \
    --event-coverage 0.45 \
    --min-duration 20 \
    --max-duration 64 \
    --min-gap 12 \
    --lead-steps 8 \
    --wind-margin-ms 1.4 \
    --cred-hysteresis-on 0.6 \
    --cred-hysteresis-off 0.3 \
    --flux-wind-exponent 3.0 \
    --event-microstructure-sigma 0.08 \
    --event-microstructure-alpha 0.22 \
    --event-microstructure-diameter-scale 0.08 \
    --event-microstructure-velocity-scale 0.20 \
    --event-particle-microstructure-correlation 0.35 \
    --event-subtypes-enabled \
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
    --event-subtype-particle-humidity-boost-pct 1.0 \
    --event-subtype-flux-wind-boost-ms 1.0 \
    --event-subtype-thermal-air-temp-drop-c 1.0 \
    --event-subtype-latent-alpha 0.22 \
    --event-subtype-particle-latent-diameter-scale-mm 0.14 \
    --event-subtype-particle-latent-velocity-scale-ms 2.4 \
    --event-subtype-flux-latent-sigma 1.2 \
    --event-subtype-thermal-latent-surface-scale-c 2.4 \
    --event-subtype-latent-target-lag-steps 4 \
    --event-subtype-context-lead-steps 8 \
    --event-subtype-context-noise-std 0.05 \
    --oracle-rollout-steps 2048 \
    --oracle-rollouts-per-policy 4 \
    --oracle-epochs 10 \
    --oracle-batch-size 512 \
    --oracle-loss-clip 20 \
    --oracle-candidate-mask-repeat 1 \
    --oracle-candidate-mask-limit 0 \
    --oracle-subtype-teacher-repeat 4 \
    --oracle-subtype-teacher-lookahead-steps 8 \
    --oracle-subtype-teacher-calm-sensors met_station_core radiometer_basic shielded_thermo_hygro \
    --oracle-subtype-teacher-particle-sensors met_station_core laser_disdrometer \
    --oracle-subtype-teacher-flux-sensors met_station_core fc4_flux \
    --oracle-subtype-teacher-thermal-sensors radiometer_basic shielded_thermo_hygro surface_temp_ir \
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
    --bc-pretrain-steps "$BC_PRETRAIN_STEPS" \
    --bc-pretrain-epochs "$BC_PRETRAIN_EPOCHS" \
    --bc-pretrain-batch-size 256 \
    --bc-pretrain-loss-coef "$BC_PRETRAIN_LOSS_COEF" \
    --subtype-aux-coef "$SUBTYPE_AUX_COEF" \
    --subtype-aux-classes 4 \
    --subtype-aux-lookahead-steps 8 \
    --no-subtype-router \
    --awbc-teacher-mode "$AWBC_TEACHER_MODE" \
    --awbc-teacher-subtype-calm-sensors met_station_core radiometer_basic shielded_thermo_hygro \
    --awbc-teacher-subtype-particle-sensors met_station_core laser_disdrometer \
    --awbc-teacher-subtype-flux-sensors met_station_core fc4_flux \
    --awbc-teacher-subtype-thermal-sensors radiometer_basic shielded_thermo_hygro surface_temp_ir \
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
