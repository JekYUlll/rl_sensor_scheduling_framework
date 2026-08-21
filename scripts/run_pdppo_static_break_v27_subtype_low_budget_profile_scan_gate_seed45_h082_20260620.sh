#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${GPU_IDS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

OUT_DIR="${OUT_DIR:-reports/v31_static_break_v27_subtype_low_budget_profile_scan_gate_seed45_h082_20260620}"
mkdir -p "$OUT_DIR"

python scripts/63_v31_static_break_calibration.py \
  --sensor-cfg configs/sensors/windblown_sensors_physical_event_v26_calm_selective.yaml \
  --out-dir "$OUT_DIR" \
  --profiles particle_heavy_flux_v7 event_flux_particle_v7 dual_flux_particle_v7 \
  --budgets 1.03 1.05 \
  --startup-peak-budgets 1.55 \
  --max-active 4 \
  --coverage-groups \
  --oracle-type tcn \
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
  --eval-steps 512 \
  --eval-rollouts 8 \
  --eval-event-fraction 0.65 \
  --env-min-dwell-steps 12 \
  --eval-start-selection event_fraction \
  --eval-selection-stride 64 \
  --schedule-family all \
  --schedule-lead-steps 6 \
  --auto-schedule-top-k 6 \
  --diverse-schedule-dwell-steps 12 \
  --deployable-static-diagnostics \
  --deployable-static-top-k 6 \
  --deployable-static-duty-low 0.12 \
  --deployable-static-duty-high 0.75 \
  --deployable-static-duty-score 12 \
  --deployable-static-duty-feedback 2.5 \
  --compare-deployable-static \
  --require-raw-static-margin \
  --target-diagnostics \
  --energy-account \
  --energy-capacity 180 \
  --initial-energy 180 \
  --harvest-per-step 0.82 \
  --reserve-energy 20 \
  --require-diverse-dynamic \
  --min-dynamic-margin 0.030 \
  --min-mid-duty-sensors 5 \
  --max-always-on-sensors 1 \
  --max-always-off-sensors 2 \
  --min-switches-per-step 0.003 \
  --max-switches-per-step 0.08 \
  --seed 45 \
  --force \
  "$@"
