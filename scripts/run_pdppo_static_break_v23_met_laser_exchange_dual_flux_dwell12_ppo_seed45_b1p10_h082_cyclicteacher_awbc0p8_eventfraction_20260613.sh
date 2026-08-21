#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/scripts" ]; then
  cd "$SCRIPT_DIR"
else
  cd "$SCRIPT_DIR/.."
fi

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate darts 2>/dev/null || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/home/zhangzhuyu/.conda/envs/darts/bin/python
if [ ! -x "$PY" ]; then
  PY="$(command -v python)"
fi

OUT_DIR="${OUT_DIR:-reports/v31_static_break_v23_met_laser_exchange_dual_flux_dwell12_ppo_seed45_b1p10_h082_cyclicteacher_awbc0p8_eventfraction_20260613}"
AWBC_COEF="${AWBC_COEF:-0.80}"
SUMMARY_NAME="${SUMMARY_NAME:-v23_metlaser_dualflux_cyclicteacher_awbc0p8_seed45_h082_eventfraction_summary.csv}"
INCLUDE_AGENT_CYCLE_PHASE="${INCLUDE_AGENT_CYCLE_PHASE:-0}"
AGENT_CYCLE_PERIOD_STEPS="${AGENT_CYCLE_PERIOD_STEPS:-0}"
AGENT_CYCLE_DWELL_STEPS="${AGENT_CYCLE_DWELL_STEPS:-1}"

CALM_TOP5="met_station_core,radiometer_basic,shielded_thermo_hygro,snow_particle_counter;met_station_core,radiometer_basic,snow_particle_counter;met_station_core,radiometer_basic,laser_disdrometer;radiometer_basic,surface_temp_ir,shielded_thermo_hygro,laser_disdrometer;radiometer_basic,shielded_thermo_hygro,snow_particle_counter"
EVENT_TOP5="met_station_core,radiometer_basic,laser_disdrometer;met_station_core,radiometer_basic,shielded_thermo_hygro,snow_particle_counter;met_station_core,radiometer_basic,snow_particle_counter;surface_temp_ir,ultrasonic_anemometer_hd,shielded_thermo_hygro,snow_particle_counter;surface_temp_ir,shielded_thermo_hygro,fc4_flux"

PHASE_ARGS=()
if [ "$INCLUDE_AGENT_CYCLE_PHASE" = "1" ]; then
  PHASE_ARGS=(
    --include-agent-cycle-phase
    --agent-cycle-period-steps "$AGENT_CYCLE_PERIOD_STEPS"
    --agent-cycle-dwell-steps "$AGENT_CYCLE_DWELL_STEPS"
  )
fi

"$PY" scripts/59_v31_split_protocol_grid.py \
  --out-dir "$OUT_DIR" \
  --sensor-cfg configs/sensors/windblown_sensors_physical_event_v23_met_laser_exchange.yaml \
  --budgets 1.10 \
  --seeds 45 \
  --workers 1 \
  --gpu-ids 5 \
  --startup-peak-budget 1.55 \
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
  --total-timesteps 40000 \
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
  "${PHASE_ARGS[@]}" \
  --awbc-coef "$AWBC_COEF" \
  --awbc-label-stride 1 \
  --awbc-teacher-mode event_cyclic \
  --awbc-teacher-calm-pool-spec "$CALM_TOP5" \
  --awbc-teacher-event-pool-spec "$EVENT_TOP5" \
  --awbc-teacher-event-lookahead-steps 6 \
  --awbc-teacher-dwell-steps 12 \
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
  --target-weights 0.03 0.03 0.10 0.01 0.01 0.0 22.0 16.0 16.0 \
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
  --budget-label budget1p10 \
  --seeds 45 \
  --out-name "$SUMMARY_NAME"
