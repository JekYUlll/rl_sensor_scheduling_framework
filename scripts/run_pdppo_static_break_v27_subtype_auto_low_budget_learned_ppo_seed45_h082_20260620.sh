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
OUT_DIR="${OUT_DIR:-reports/v31_static_break_v27_subtype_auto_low_budget_${PROFILE_NAME}_b${BUDGET_TAG}_learned_ppo_seed45_h082_20260620}"
SUMMARY_NAME="${SUMMARY_NAME:-v27_subtype_auto_${PROFILE_NAME}_b${BUDGET_TAG}_learned_ppo_seed45_h082_summary.csv}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-80000}"
GPU_IDS="${GPU_IDS:-5}"
SEEDS="${SEEDS:-45}"
WORKERS="${WORKERS:-1}"

AWBC_COEF="${AWBC_COEF:-0.80}"
AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS="${AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS:-0}"
AWBC_TEACHER_DWELL_STEPS="${AWBC_TEACHER_DWELL_STEPS:-1}"
BC_PRETRAIN_STEPS="${BC_PRETRAIN_STEPS:-0}"
BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS:-4}"
BC_PRETRAIN_BATCH_SIZE="${BC_PRETRAIN_BATCH_SIZE:-128}"
BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-1.0}"
SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-0.0}"
SUBTYPE_AUX_CLASSES="${SUBTYPE_AUX_CLASSES:-4}"
SUBTYPE_AUX_LOOKAHEAD_STEPS="${SUBTYPE_AUX_LOOKAHEAD_STEPS:-0}"
INCLUDE_OBSERVABLE_REGIME_BELIEF="${INCLUDE_OBSERVABLE_REGIME_BELIEF:-0}"
REGIME_BELIEF_LOOKBACK="${REGIME_BELIEF_LOOKBACK:-6}"
ENT_COEF="${ENT_COEF:-0.002}"
PRIOR_KL_COEF="${PRIOR_KL_COEF:-0.03}"
LAMBDA_DUTY_BALANCE="${LAMBDA_DUTY_BALANCE:-0.8}"
DUTY_SCORE_FEEDBACK="${DUTY_SCORE_FEEDBACK:-2.5}"
REGIME_BELIEF_FLAGS=()
if [ "$INCLUDE_OBSERVABLE_REGIME_BELIEF" = "1" ]; then
  REGIME_BELIEF_FLAGS=(--include-observable-regime-belief)
fi

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

"$PY" scripts/59_v31_split_protocol_grid.py \
  --out-dir "$OUT_DIR" \
  --sensor-cfg "$SENSOR_CFG" \
  --budgets "$BUDGET" \
  --seeds $SEEDS \
  --workers "$WORKERS" \
  --gpu-ids "$GPU_IDS" \
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
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --n-steps 512 \
  --batch-size 64 \
  --n-epochs 10 \
  --train-episode-len 512 \
  --event-gated-actor \
  "${REGIME_BELIEF_FLAGS[@]}" \
  --regime-belief-lookback "$REGIME_BELIEF_LOOKBACK" \
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
  --lambda-duty-balance "$LAMBDA_DUTY_BALANCE" \
  --duty-balance-low 0.12 \
  --duty-balance-high 0.75 \
  --duty-score-feedback "$DUTY_SCORE_FEEDBACK" \
  --duty-hard-guard \
  --duty-hard-low 0.12 \
  --duty-hard-high 0.75 \
  --duty-hard-score 12 \
  --min-dwell-steps 12 \
  --awbc-coef "$AWBC_COEF" \
  --awbc-label-stride 1 \
  --bc-pretrain-steps "$BC_PRETRAIN_STEPS" \
  --bc-pretrain-epochs "$BC_PRETRAIN_EPOCHS" \
  --bc-pretrain-batch-size "$BC_PRETRAIN_BATCH_SIZE" \
  --bc-pretrain-loss-coef "$BC_PRETRAIN_LOSS_COEF" \
  --subtype-aux-coef "$SUBTYPE_AUX_COEF" \
  --subtype-aux-classes "$SUBTYPE_AUX_CLASSES" \
  --subtype-aux-lookahead-steps "$SUBTYPE_AUX_LOOKAHEAD_STEPS" \
  --awbc-teacher-mode subtype_auto \
  --awbc-teacher-event-lookahead-steps "$AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS" \
  --awbc-teacher-dwell-steps "$AWBC_TEACHER_DWELL_STEPS" \
  --awbc-teacher-subtype-calm-sensors met_station_core surface_temp_ir snow_particle_counter \
  --awbc-teacher-subtype-particle-sensors radiometer_basic shielded_thermo_hygro laser_disdrometer \
  --awbc-teacher-subtype-flux-sensors surface_temp_ir shielded_thermo_hygro laser_disdrometer \
  --awbc-teacher-subtype-thermal-sensors surface_temp_ir shielded_thermo_hygro laser_disdrometer \
  --prior-kl-coef "$PRIOR_KL_COEF" \
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
  --ent-coef "$ENT_COEF" \
  --target-weights $TARGET_WEIGHTS \
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

"$PY" scripts/65_v31_collect_operational_pdppo.py \
  --base-dir "$OUT_DIR" \
  --budget-label "budget${BUDGET_TAG}" \
  --seeds $SEEDS \
  --out-name "$SUMMARY_NAME"

mapfile -t CUSTOM_ROLLOUTS < <(find "$OUT_DIR/raw" -path "*/rollout_custom_ppo.npz" -type f | sort)
if [ "${#CUSTOM_ROLLOUTS[@]}" -gt 0 ]; then
  "$PY" scripts/71_v31_behavior_complexity_audit.py \
    "${CUSTOM_ROLLOUTS[@]}" \
    --out-dir "$OUT_DIR/behavior_audit"
else
  echo "No rollout_custom_ppo.npz files found for behavior audit under $OUT_DIR/raw" >&2
fi
