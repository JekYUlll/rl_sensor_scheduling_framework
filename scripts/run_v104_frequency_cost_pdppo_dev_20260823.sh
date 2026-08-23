#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-1301 1302 1303 1304 1305}"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2 3 4}"
POLICY_SEED_OFFSET="${POLICY_SEED_OFFSET:-3000}"
SCENE_PREFIX=v103_frequency_cost_scene_gate_dev
POLICY_PREFIX="${RUN_PREFIX_OVERRIDE:-v104_frequency_cost_pdppo_dev}"
CONTEXT_OUT=reports/aggregate/v103_frequency_cost_context_gate_20260823
SENSOR_CFG=configs/sensors/windblown_sensors_flexible_subset_v5_frequency_cost.yaml
mkdir -p logs/v104_frequency_cost_pdppo

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
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
    export POLICY_SEED="$((seed + POLICY_SEED_OFFSET))"
    export SENSOR_CFG TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS_OVERRIDE:-40960}" TRUTH_STEPS=36000 LOOKBACK=20
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
    export BC_PRETRAIN_STEPS="${BC_PRETRAIN_STEPS_OVERRIDE:-2000}"
    export BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS_OVERRIDE:-12}"
    export BC_PRETRAIN_LOSS_COEF=1.0
    export BC_PRETRAIN_TARGET_MODE="${BC_PRETRAIN_TARGET_MODE_OVERRIDE:-hard}"
    export BC_SOFT_TEMPERATURE="${BC_SOFT_TEMPERATURE_OVERRIDE:-1.0}"
    export AWBC_COEF="${AWBC_COEF_OVERRIDE:-0.05}" AWBC_DECAY_TIMESTEPS=0 AWBC_EVENT_ONLY=0
    export AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE_OVERRIDE:-subtype_static_auto}"
    export GREEDY_LOOKAHEAD_STEPS="${GREEDY_LOOKAHEAD_STEPS_OVERRIDE:-4}"
    export SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF_OVERRIDE:-0.3}"
    export SUBTYPE_ACTION_CE_COEF="${SUBTYPE_ACTION_CE_COEF_OVERRIDE:-0.05}"
    export SUBTYPE_ACTION_EVENT_ONLY=0
    export SUBTYPE_ACTION_SUPERVISION_MODE=positive_sensor_inclusion
    export SUBTYPE_LOSS_WEIGHTING=1
    export CONTEXT_FEATURE_DIM=20 CONTEXT_FUSION_MODE=gated_add
    export TEMPORAL_ENCODER="${TEMPORAL_ENCODER_OVERRIDE:-0}"
    export TEMPORAL_HIDDEN_DIM="${TEMPORAL_HIDDEN_DIM_OVERRIDE:-64}"
    export TRAINABLE_ACTION_PRIOR=0 NONLINEAR_ACTION_EMBEDDING=1
    export ENT_COEF=0.02 CHANNEL_MARGINAL_ENTROPY_COEF=0
    export CHECKPOINT_SELECTION_INTERVAL_UPDATES="${CHECKPOINT_SELECTION_INTERVAL_UPDATES:-0}"
    export POLICY_CHECKPOINT_SOURCE="${POLICY_CHECKPOINT_SOURCE:-}"
    export EVALUATION_POLICY_MODE="${EVALUATION_POLICY_MODE:-deterministic}"
    export EVALUATION_TEMPERATURE_CANDIDATES="${EVALUATION_TEMPERATURE_CANDIDATES:-}"
    bash scripts/run_v32_flexible_subset_pilot_20260822.sh "$seed"
  ) >"logs/v104_frequency_cost_pdppo/seed${seed}.log" 2>&1 &
done
wait
