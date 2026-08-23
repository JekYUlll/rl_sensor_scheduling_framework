#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
SEEDS=(1301 1302 1303 1304 1305)
SCENE_PREFIX=v103_frequency_cost_scene_gate_dev
POLICY_PREFIX=v104_frequency_cost_pdppo_dev
CONTEXT_OUT=reports/aggregate/v103_frequency_cost_context_gate_20260823
SENSOR_CFG=configs/sensors/windblown_sensors_flexible_subset_v5_frequency_cost.yaml
mkdir -p logs/v104_frequency_cost_pdppo

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="$i"
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
    export SENSOR_CFG TOTAL_TIMESTEPS=40960 TRUTH_STEPS=36000 LOOKBACK=20
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
    export BC_PRETRAIN_STEPS=2000 BC_PRETRAIN_EPOCHS=12 BC_PRETRAIN_LOSS_COEF=1.0
    export AWBC_COEF=0.05 AWBC_DECAY_TIMESTEPS=0 AWBC_EVENT_ONLY=0
    export AWBC_TEACHER_MODE=subtype_static_auto
    export SUBTYPE_AUX_COEF=0.3 SUBTYPE_ACTION_CE_COEF=0.05
    export SUBTYPE_ACTION_EVENT_ONLY=0
    export SUBTYPE_ACTION_SUPERVISION_MODE=positive_sensor_inclusion
    export SUBTYPE_LOSS_WEIGHTING=1
    export CONTEXT_FEATURE_DIM=20 CONTEXT_FUSION_MODE=gated_add
    export TRAINABLE_ACTION_PRIOR=0 NONLINEAR_ACTION_EMBEDDING=1
    export ENT_COEF=0.02 CHANNEL_MARGINAL_ENTROPY_COEF=0
    export EVALUATION_POLICY_MODE=deterministic
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/v104_frequency_cost_pdppo/seed${seed}.log" 2>&1 &
done
wait
