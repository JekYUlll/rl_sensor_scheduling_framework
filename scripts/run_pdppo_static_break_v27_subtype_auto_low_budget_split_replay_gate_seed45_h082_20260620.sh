#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate darts 2>/dev/null || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python)"
fi

PROFILE_NAME="${PROFILE_NAME:-particle_heavy_flux_v7}"
BUDGET="${BUDGET:-1.05}"
BUDGET_TAG="${BUDGET_TAG:-1p05}"
STARTUP_PEAK_BUDGET="${STARTUP_PEAK_BUDGET:-1.55}"
SENSOR_CFG="${SENSOR_CFG:-configs/sensors/windblown_sensors_physical_event_v26_calm_selective.yaml}"

if [ -z "${TARGET_WEIGHTS:-}" ]; then
  case "$PROFILE_NAME" in
    particle_heavy_flux_v7)
      TARGET_WEIGHTS="0.03 0.03 0.10 0.01 0.01 0.0 16.0 22.0 22.0"
      ;;
    event_flux_particle_v7)
      TARGET_WEIGHTS="0.03 0.03 0.10 0.01 0.01 0.0 30.0 12.0 12.0"
      ;;
    dual_flux_particle_v7)
      TARGET_WEIGHTS="0.03 0.03 0.10 0.01 0.01 0.0 22.0 16.0 16.0"
      ;;
    *)
      echo "Unknown PROFILE_NAME=$PROFILE_NAME; set TARGET_WEIGHTS explicitly." >&2
      exit 2
      ;;
  esac
fi

SOURCE_DIR="${SOURCE_DIR:-reports/v31_static_break_v27_subtype_auto_low_budget_${PROFILE_NAME}_b${BUDGET_TAG}_zero_ppo_source_seed45_h082_20260620}"
REPLAY_DIR="${REPLAY_DIR:-reports/v31_static_break_v27_subtype_auto_low_budget_${PROFILE_NAME}_b${BUDGET_TAG}_split_replay_gate_seed45_h082_20260620}"

mkdir -p "$SOURCE_DIR" "$REPLAY_DIR"

"$PY" scripts/59_v31_split_protocol_grid.py \
  --out-dir "$SOURCE_DIR" \
  --sensor-cfg "$SENSOR_CFG" \
  --budgets "$BUDGET" \
  --seeds 45 \
  --workers 1 \
  --gpu-ids "${GPU_IDS:-5}" \
  --startup-peak-budget "$STARTUP_PEAK_BUDGET" \
  --truth-steps 60000 \
  --freq-s 3600 \
  --event-coverage 0.55 \
  --min-duration 12 \
  --max-duration 36 \
  --min-gap 2 \
  --lead-steps 6 \
  --flux-wind-exponent 4.0 \
  --event-microstructure-sigma 0.65 \
  --event-microstructure-alpha 0.20 \
  --event-microstructure-diameter-scale 0.16 \
  --event-microstructure-velocity-scale 1.50 \
  --event-particle-microstructure-correlation 0.00 \
  --event-subtypes-enabled \
  --event-subtype-particle-prob 0.34 \
  --event-subtype-flux-prob 0.33 \
  --event-subtype-thermal-prob 0.33 \
  --event-subtype-particle-flux-multiplier 0.65 \
  --event-subtype-flux-multiplier 2.80 \
  --event-subtype-thermal-flux-multiplier 0.45 \
  --event-subtype-particle-diameter-shift-mm 0.12 \
  --event-subtype-particle-velocity-boost-ms 1.50 \
  --event-subtype-flux-diameter-shift-mm -0.05 \
  --event-subtype-flux-velocity-boost-ms 0.80 \
  --event-subtype-thermal-surface-drop-c 2.40 \
  --oracle-rollout-steps 2400 \
  --oracle-rollouts-per-policy 4 \
  --oracle-epochs 8 \
  --oracle-batch-size 512 \
  --oracle-device auto \
  --oracle-inference-device cpu \
  --total-timesteps 0 \
  --n-steps 512 \
  --batch-size 64 \
  --n-epochs 10 \
  --train-episode-len 512 \
  --event-gated-actor \
  --event-start-prob 0.65 \
  --event-reward-multiplier 1.5 \
  --soc-aux-horizon 32 \
  --soc-aux-coef 0.03 \
  --lambda-warmup-abort 1.00 \
  --lambda-switch 0.002 \
  --energy-account \
  --energy-capacity 180 \
  --initial-energy 180 \
  --harvest-per-step 0.82 \
  --reserve-energy 20 \
  --soc-soft-penalty-buffer 40 \
  --lambda-soc-soft-penalty 0.08 \
  --lambda-duty-balance 0.8 \
  --duty-balance-low 0.12 \
  --duty-balance-high 0.75 \
  --duty-score-feedback 2.5 \
  --duty-hard-guard \
  --duty-hard-low 0.12 \
  --duty-hard-high 0.75 \
  --duty-hard-score 12 \
  --min-dwell-steps 12 \
  --awbc-coef 0.40 \
  --awbc-label-stride 1 \
  --awbc-teacher-mode oracle_greedy \
  --prior-kl-coef 0.05 \
  --greedy-lookahead-steps 4 \
  --use-candidate-prior \
  --candidate-prior-scale 0.5 \
  --candidate-prior-steps 512 \
  --candidate-prior-rollouts 4 \
  --static-selection-steps 512 \
  --static-selection-rollouts 4 \
  --eval-steps 512 \
  --eval-rollouts 8 \
  --eval-start-selection event_fraction \
  --eval-event-fraction 0.65 \
  --eval-selection-stride 64 \
  --ent-coef 0.001 \
  --target-weights ${TARGET_WEIGHTS} \
  --target-scales 5.0 5.0 5.0 1.0 1.0 100.0 0.0001 0.2 5.0 \
  --max-active 4 \
  --eval-duty-constrained-baselines \
  --baseline-duty-hard-low 0.12 \
  --baseline-duty-hard-high 0.75 \
  --baseline-duty-hard-score 12 \
  --baseline-duty-score-feedback 2.5 \
  --bonferroni-family 1 \
  --skip-rollout-evaluation \
  --skip-collect

"$PY" scripts/70_v31_split_replay_gate.py \
  --source-run-dir "$SOURCE_DIR/raw/budget${BUDGET_TAG}_seed45" \
  --out-dir "$REPLAY_DIR" \
  --oracle-device cpu \
  --env-min-dwell-steps 12 \
  --replay-family subtype_auto \
  --top-sizes 2 \
  --subtype-top-size-cap 2 \
  --lead-steps 0 6 \
  --dwell-steps 1 \
  --min-margin-abs 0.005 \
  --min-margin-rel 0.01
