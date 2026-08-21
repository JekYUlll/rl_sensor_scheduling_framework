#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate darts 2>/dev/null || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python)"
fi

OUT_DIR="${OUT_DIR:-reports/v31_static_break_v26_calm_selective_low_budget_profile_scan_dwell12_gate_seed45_h082_20260620}"
mkdir -p "$OUT_DIR"

# V26 framework/sensor-model gate:
# add calm/non-event observation unreliability for event instruments, then run
# the same low-budget profile scan under a strict raw-static margin requirement.
"$PY" scripts/63_v31_static_break_calibration.py \
  --sensor-cfg configs/sensors/windblown_sensors_physical_event_v26_calm_selective.yaml \
  --out-dir "$OUT_DIR" \
  --profiles particle_heavy_flux_v7 event_flux_particle_v7 dual_flux_particle_v7 \
  --budgets 1.03 1.05 1.08 \
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
  --force
