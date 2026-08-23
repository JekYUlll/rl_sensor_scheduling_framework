#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
PHASE="${1:-all}"
SEEDS=(1301 1302 1303 1304 1305)
PREFIX=v103_frequency_cost_scene_gate_dev
CONTEXT_OUT=reports/aggregate/v103_frequency_cost_context_gate_20260823
SENSOR_CFG=configs/sensors/windblown_sensors_flexible_subset_v5_frequency_cost.yaml

run_parallel() {
  local worker="$1" i seed
  for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    "$worker" "$seed" "$i" &
  done
  wait
}

common_scene_env() {
  export SENSOR_CFG TOTAL_TIMESTEPS=1024 TRUTH_STEPS=36000 LOOKBACK=20
  export EVENT_COVERAGE=0.55 MIN_DURATION=20 MAX_DURATION=64 MIN_GAP=12
  export EVENT_MICROSTRUCTURE_SIGMA=0.12 EVENT_MICROSTRUCTURE_ALPHA=0.15
  export EVENT_PARTICLE_MICROSTRUCTURE_CORRELATION=0.0
  export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY=0.8
  export EVENT_SUBTYPE_ASSIGNMENT=stratified EVENT_SUBTYPE_LATENT_ALPHA=0.15
  export PARTICLE_LATENT_DIAMETER_SCALE=0.28 PARTICLE_LATENT_VELOCITY_SCALE=4.8
  export FLUX_LATENT_SIGMA=2.0 THERMAL_LATENT_SURFACE_SCALE=2.4
  export EVENT_SUBTYPE_CONTEXT_LEAD_STEPS=12 EVENT_SUBTYPE_CONTEXT_NOISE_STD=0.02
  export ORACLE_EPOCHS=10 ORACLE_FULL_OPEN_REPEAT=3 ORACLE_CANDIDATE_MASK_REPEAT=2
  export ORACLE_SUBTYPE_TEACHER_REPEAT=6 ORACLE_INFERENCE_DEVICE=cpu
  export BUDGET=1.75 STARTUP_BUDGET=2.15 BUDGET_LABEL=b1p75
  export TARGET_WEIGHTS='1 1 1 1 1 1 1 1 1'
  export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=1 BC_PRETRAIN_LOSS_COEF=0.0
  export AWBC_COEF=0.0 SUBTYPE_AUX_COEF=0.0 SUBTYPE_ACTION_CE_COEF=0.0
  export SUBTYPE_ACTION_SUPERVISION_MODE=positive_sensor_inclusion
  export SUBTYPE_LOSS_WEIGHTING=1 CONTEXT_FEATURE_DIM=20 CONTEXT_FUSION_MODE=gated_add
  export TRAINABLE_ACTION_PRIOR=0 NONLINEAR_ACTION_EMBEDDING=1
}

run_scene() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu" RUN_PREFIX="$PREFIX"
    common_scene_env
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/v103_scene_seed${seed}.log" 2>&1
}

run_context() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" \
      scripts/81_v31_framework_baseline_supplements.py \
      --run-glob "reports/${PREFIX}_seed${seed}_b1p75_20260822" \
      --seeds "$seed" --out-root "$CONTEXT_OUT" --router-eval-dir . \
      --replay-dir __none__ --oracle-device cuda --policies context_bandit event_label \
      --context-thresholds 0.5 --context-action-source replay_calibrated \
      --greedy-max-steps 0 --no-aggregate
  ) >"logs/v103_context_seed${seed}.log" 2>&1
}

run_receding() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" scripts/99_v32_receding_upper.py \
      --run-dir "reports/${PREFIX}_seed${seed}_b1p75_20260822" --device cuda
  ) >"logs/v103_receding_seed${seed}.log" 2>&1
}

aggregate_context() {
  env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" \
    scripts/81_v31_framework_baseline_supplements.py \
    --run-glob "reports/${PREFIX}_seed*_b1p75_20260822" \
    --seeds "${SEEDS[@]}" --out-root "$CONTEXT_OUT" --router-eval-dir . \
    --replay-dir __none__ --oracle-device cpu --policies context_bandit event_label \
    --context-thresholds 0.5 --context-action-source replay_calibrated \
    --greedy-max-steps 0 --reuse-existing-seed-metrics \
    >logs/v103_context_aggregate.log 2>&1
}

mkdir -p logs "$CONTEXT_OUT"
case "$PHASE" in
  scene) run_parallel run_scene ;;
  context) run_parallel run_context; aggregate_context ;;
  receding) run_parallel run_receding ;;
  all) run_parallel run_scene; run_parallel run_context; aggregate_context; run_parallel run_receding ;;
  *) printf 'unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
esac
