#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
PHASE="${1:-all}"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2 3 4}"
MAX_GPUS="${MAX_GPUS:-${#GPU_LIST[@]}}"
if ((MAX_GPUS < 1 || MAX_GPUS > ${#GPU_LIST[@]})); then
  printf 'MAX_GPUS must be between 1 and the number of GPU_IDS\n' >&2
  exit 2
fi
SEEDS=(1201 1202 1203 1204 1205 1206 1207 1208 1209 1210 1211 1212 1213 1214 1215 1216 1217 1218 1219 1220 1221 1222)
SCENE_PREFIX=v102_v97_frozen_scene_final
POLICY_PREFIX=v102_v97_frozen_pdppo_final
CONTEXT_OUT=reports/aggregate/v102_v97_frozen_context_final_20260823
SENSOR_CFG=configs/sensors/windblown_sensors_flexible_subset_v4_specificity.yaml

run_batches() {
  local worker="$1"
  local start i slot seed
  for ((start=0; start<${#SEEDS[@]}; start+=MAX_GPUS)); do
    for ((slot=0; slot<MAX_GPUS && start+slot<${#SEEDS[@]}; slot++)); do
      i=$((start + slot))
      seed="${SEEDS[$i]}"
      "$worker" "$seed" "${GPU_LIST[$slot]}" &
    done
    wait
  done
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
  export SUBTYPE_LOSS_WEIGHTING=1 CONTEXT_FEATURE_DIM=20 CONTEXT_FUSION_MODE=gated_add
  export TRAINABLE_ACTION_PRIOR=0 NONLINEAR_ACTION_EMBEDDING=1
}

run_scene() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu" RUN_PREFIX="$SCENE_PREFIX"
    common_scene_env
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=1 BC_PRETRAIN_LOSS_COEF=0.0
    export AWBC_COEF=0.0 SUBTYPE_AUX_COEF=0.0 SUBTYPE_ACTION_CE_COEF=0.0
    export SUBTYPE_ACTION_SUPERVISION_MODE=positive_sensor_inclusion
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/v102_scene_seed${seed}.log" 2>&1
}

run_context() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" \
      scripts/81_v31_framework_baseline_supplements.py \
      --run-glob "reports/${SCENE_PREFIX}_seed${seed}_b1p75_20260822" \
      --seeds "$seed" --out-root "$CONTEXT_OUT" --router-eval-dir . \
      --replay-dir __none__ --oracle-device cuda --policies context_bandit event_label \
      --context-thresholds 0.5 --context-action-source replay_calibrated \
      --greedy-max-steps 0 --no-aggregate
  ) >"logs/v102_context_seed${seed}.log" 2>&1
}

run_policy() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    mapfile -t teacher_masks < <("$PY" - "$seed" "$CONTEXT_OUT" "$SCENE_PREFIX" <<'PY'
import csv
import json
import sys
from pathlib import Path

seed, context_out, scene_prefix = int(sys.argv[1]), sys.argv[2], sys.argv[3]
row = next(csv.DictReader(open(f"{context_out}/seed{seed}/context_bandit_action_map.csv")))
geometry = json.loads(Path(f"reports/{scene_prefix}_seed{seed}_b1p75_20260822/action_geometry.json").read_text())
for label in ("calm", "particle", "flux", "thermal"):
    print(" ".join(geometry["masks"][int(row[label])]["sensor_ids"]))
PY
    )
    export AWBC_TEACHER_CALM_SENSORS="${teacher_masks[0]}"
    export AWBC_TEACHER_PARTICLE_SENSORS="${teacher_masks[1]}"
    export AWBC_TEACHER_FLUX_SENSORS="${teacher_masks[2]}"
    export AWBC_TEACHER_THERMAL_SENSORS="${teacher_masks[3]}"
    export RUN_PREFIX="$POLICY_PREFIX"
    export CONTROL_SOURCE_RUN_DIR="reports/${SCENE_PREFIX}_seed${seed}_b1p75_20260822"
    export POLICY_SEED="$((seed + 3000))"
    common_scene_env
    export TOTAL_TIMESTEPS=40960
    export BC_PRETRAIN_STEPS=2000 BC_PRETRAIN_EPOCHS=12 BC_PRETRAIN_LOSS_COEF=1.0
    export AWBC_COEF=0.05 AWBC_DECAY_TIMESTEPS=0 AWBC_EVENT_ONLY=0
    export AWBC_TEACHER_MODE=subtype_static_auto
    export SUBTYPE_AUX_COEF=0.3 SUBTYPE_ACTION_CE_COEF=0.05
    export SUBTYPE_ACTION_EVENT_ONLY=0
    export SUBTYPE_ACTION_SUPERVISION_MODE=positive_sensor_inclusion
    export ENT_COEF=0.02 CHANNEL_MARGINAL_ENTROPY_COEF=0
    export EVALUATION_POLICY_MODE=deterministic
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/v102_policy_seed${seed}.log" 2>&1
}

mkdir -p logs "$CONTEXT_OUT"
case "$PHASE" in
  scene) run_batches run_scene ;;
  context)
    run_batches run_context
    env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" \
      scripts/81_v31_framework_baseline_supplements.py \
      --run-glob "reports/${SCENE_PREFIX}_seed*_b1p75_20260822" \
      --seeds "${SEEDS[@]}" --out-root "$CONTEXT_OUT" --router-eval-dir . \
      --replay-dir __none__ --oracle-device cpu --policies context_bandit event_label \
      --context-thresholds 0.5 --context-action-source replay_calibrated \
      --greedy-max-steps 0 --reuse-existing-seed-metrics \
      >logs/v102_context_aggregate.log 2>&1
    ;;
  policy) run_batches run_policy ;;
  all)
    run_batches run_scene
    "$0" context
    run_batches run_policy
    ;;
  *) printf 'unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
esac
