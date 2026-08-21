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
BUDGET="${BUDGET:-1.03}"
BUDGET_TAG="${BUDGET_TAG:-1p03}"
STARTUP_PEAK_BUDGET="${STARTUP_PEAK_BUDGET:-1.55}"
SENSOR_CFG="${SENSOR_CFG:-configs/sensors/windblown_sensors_physical_event_v24_event_selective_laser.yaml}"
OUT_DIR="${OUT_DIR:-reports/v31_static_break_v25_v24_low_budget_${PROFILE_NAME}_b${BUDGET_TAG}_learned_ppo_seed45_h082_20260620}"
SUMMARY_NAME="${SUMMARY_NAME:-v25_lowbudget_${PROFILE_NAME}_b${BUDGET_TAG}_learned_ppo_seed45_h082_summary.csv}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000}"
GPU_IDS="${GPU_IDS:-4}"
SEEDS="${SEEDS:-45}"
WORKERS="${WORKERS:-1}"

AWBC_COEF="${AWBC_COEF:-0.40}"
AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-oracle_greedy}"
AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS="${AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS:-6}"
AWBC_TEACHER_DWELL_STEPS="${AWBC_TEACHER_DWELL_STEPS:-12}"
AWBC_TEACHER_CALM_POOL_SPEC="${AWBC_TEACHER_CALM_POOL_SPEC:-}"
AWBC_TEACHER_EVENT_POOL_SPEC="${AWBC_TEACHER_EVENT_POOL_SPEC:-}"

INCLUDE_AGENT_CYCLE_PHASE="${INCLUDE_AGENT_CYCLE_PHASE:-0}"
AGENT_CYCLE_PERIOD_STEPS="${AGENT_CYCLE_PERIOD_STEPS:-0}"
AGENT_CYCLE_DWELL_STEPS="${AGENT_CYCLE_DWELL_STEPS:-1}"

ENT_COEF="${ENT_COEF:-0.003}"
PRIOR_KL_COEF="${PRIOR_KL_COEF:-0.05}"
LAMBDA_DUTY_BALANCE="${LAMBDA_DUTY_BALANCE:-0.8}"
DUTY_SCORE_FEEDBACK="${DUTY_SCORE_FEEDBACK:-2.5}"

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

PHASE_ARGS=()
if [ "$INCLUDE_AGENT_CYCLE_PHASE" = "1" ]; then
  PHASE_ARGS=(
    --include-agent-cycle-phase
    --agent-cycle-period-steps "$AGENT_CYCLE_PERIOD_STEPS"
    --agent-cycle-dwell-steps "$AGENT_CYCLE_DWELL_STEPS"
  )
fi

TEACHER_POOL_ARGS=()
if [ -n "$AWBC_TEACHER_CALM_POOL_SPEC" ]; then
  TEACHER_POOL_ARGS+=(--awbc-teacher-calm-pool-spec "$AWBC_TEACHER_CALM_POOL_SPEC")
fi
if [ -n "$AWBC_TEACHER_EVENT_POOL_SPEC" ]; then
  TEACHER_POOL_ARGS+=(--awbc-teacher-event-pool-spec "$AWBC_TEACHER_EVENT_POOL_SPEC")
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
  "${PHASE_ARGS[@]}" \
  --awbc-coef "$AWBC_COEF" \
  --awbc-label-stride 1 \
  --awbc-teacher-mode "$AWBC_TEACHER_MODE" \
  --awbc-teacher-event-lookahead-steps "$AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS" \
  --awbc-teacher-dwell-steps "$AWBC_TEACHER_DWELL_STEPS" \
  "${TEACHER_POOL_ARGS[@]}" \
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
