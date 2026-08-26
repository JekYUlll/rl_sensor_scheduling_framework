#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
PHASE="${1:-all}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-1501 1502 1503 1504 1505}"
PREFIX="${RUN_PREFIX_OVERRIDE:-v137_generic_physical_scene_gate_dev}"
CONTEXT_OUT="${CONTEXT_OUT_OVERRIDE:-reports/aggregate/v137_generic_physical_context_gate_20260826}"
SENSOR_CFG=configs/sensors/windblown_sensors_flexible_subset_v6_physical_channels.yaml

run_parallel() {
  local worker="$1" index seed
  for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    "$worker" "$seed" "$index" &
  done
  wait
}

scene_env() {
  export SENSOR_CFG TOTAL_TIMESTEPS=1024 TRUTH_STEPS=36000 LOOKBACK=20
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
  export ORACLE_SUBTYPE_TEACHER_REPEAT=0 ORACLE_INFERENCE_DEVICE=cpu
  export BUDGET=1.75 STARTUP_BUDGET=2.15 BUDGET_LABEL=b1p75
  export TARGET_WEIGHTS='1 1 1 1 1 1 1 1 1'
  export CHANNEL_QUALITY_ENABLED="${CHANNEL_QUALITY_ENABLED_OVERRIDE:-0}"
  export CHANNEL_QUALITY_DEGRADED_COVERAGE="${CHANNEL_QUALITY_DEGRADED_COVERAGE_OVERRIDE:-0.0}"
  export CHANNEL_QUALITY_MIN_DURATION_STEPS="${CHANNEL_QUALITY_MIN_DURATION_STEPS_OVERRIDE:-12}"
  export CHANNEL_QUALITY_MAX_DURATION_STEPS="${CHANNEL_QUALITY_MAX_DURATION_STEPS_OVERRIDE:-48}"
  export CHANNEL_QUALITY_MIN_GAP_STEPS="${CHANNEL_QUALITY_MIN_GAP_STEPS_OVERRIDE:-12}"
  export CHANNEL_QUALITY_DEGRADED_VALUE="${CHANNEL_QUALITY_DEGRADED_VALUE_OVERRIDE:-0.2}"
  export CHANNEL_QUALITY_TRANSITION_STEPS="${CHANNEL_QUALITY_TRANSITION_STEPS_OVERRIDE:-0}"
  export CHANNEL_QUALITY_REPORT_NOISE_STD="${CHANNEL_QUALITY_REPORT_NOISE_STD_OVERRIDE:-0.02}"
  export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER="${SENSOR_QUALITY_MAX_NOISE_MULTIPLIER_OVERRIDE:-1.0}"
  export SENSOR_QUALITY_AVAILABILITY_FLOOR="${SENSOR_QUALITY_AVAILABILITY_FLOOR_OVERRIDE:-1.0}"
  export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=1 BC_PRETRAIN_LOSS_COEF=0
  export AWBC_COEF=0 SUBTYPE_AUX_COEF=0 SUBTYPE_ACTION_CE_COEF=0
  export SUBTYPE_LOSS_WEIGHTING=0 CONTEXT_FEATURE_DIM=20 CONTEXT_FUSION_MODE=gated_add
  export TRAINABLE_ACTION_PRIOR=0 NONLINEAR_ACTION_EMBEDDING=1
}

run_scene() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu" RUN_PREFIX="$PREFIX"
    scene_env
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/v137_scene_seed${seed}.log" 2>&1
}

run_context() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" \
      scripts/81_v31_framework_baseline_supplements.py \
      --run-glob "reports/${PREFIX}_seed${seed}_b1p75_20260822" \
      --seeds "$seed" --out-root "$CONTEXT_OUT" --router-eval-dir . \
      --replay-dir __none__ --oracle-device cuda \
      --policies context_bandit forecast_greedy \
      --context-thresholds 0.5 --context-action-source replay_calibrated \
      --greedy-max-steps 0 --no-aggregate
  ) >"logs/v137_context_seed${seed}.log" 2>&1
}

run_receding() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" scripts/99_v32_receding_upper.py \
      --run-dir "reports/${PREFIX}_seed${seed}_b1p75_20260822" --device cuda
  ) >"logs/v137_receding_seed${seed}.log" 2>&1
}

mkdir -p logs "$CONTEXT_OUT"
case "$PHASE" in
  scene) run_parallel run_scene ;;
  context) run_parallel run_context ;;
  receding) run_parallel run_receding ;;
  all) run_parallel run_scene; run_parallel run_context; run_parallel run_receding ;;
  *) printf 'unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
esac
