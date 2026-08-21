#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate darts 2>/dev/null || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

OUT_ROOT="reports/v31_static_break_v16_multiseed_structural_screen_20260609"
mkdir -p "$OUT_ROOT"

SEEDS=(41 42 43)
PROFILES=(
  micro_flux_v6
  micro_particle_v6
  flux_micro_v6
  dual_flux_particle_v7
  event_flux_particle_v7
  particle_heavy_flux_v7
)

for seed in "${SEEDS[@]}"; do
  OUT_DIR="$OUT_ROOT/seed${seed}"
  mkdir -p "$OUT_DIR"
  python scripts/63_v31_static_break_calibration.py \
    --sensor-cfg configs/sensors/windblown_sensors_physical_event_v16_surface_boundary.yaml \
    --out-dir "$OUT_DIR" \
    --profiles "${PROFILES[@]}" \
    --budgets 1.15 \
    --startup-peak-budgets 1.55 \
    --max-active 4 \
    --coverage-groups \
    --oracle-type tcn \
    --truth-steps 60000 \
    --freq-s 3600 \
    --event-coverage 0.34 \
    --min-duration 12 \
    --max-duration 28 \
    --min-gap 4 \
    --lead-steps 6 \
    --flux-wind-exponent 4.0 \
    --event-microstructure-sigma 0.45 \
    --event-microstructure-alpha 0.20 \
    --event-microstructure-diameter-scale 0.08 \
    --event-microstructure-velocity-scale 1.00 \
    --event-particle-microstructure-correlation 0.20 \
    --oracle-rollout-steps 1800 \
    --oracle-rollouts-per-policy 4 \
    --oracle-epochs 6 \
    --oracle-batch-size 512 \
    --oracle-device auto \
    --oracle-inference-device cpu \
    --eval-steps 512 \
    --eval-rollouts 4 \
    --eval-event-fraction 0.65 \
    --env-min-dwell-steps 12 \
    --eval-start-selection event_transport_rich \
    --eval-selection-stride 64 \
    --schedule-family all \
    --schedule-lead-steps 6 \
    --auto-schedule-top-k 6 \
    --diverse-schedule-dwell-steps 12 \
    --deployable-static-diagnostics \
    --deployable-static-top-k 8 \
    --deployable-static-duty-low 0.12 \
    --deployable-static-duty-high 0.75 \
    --deployable-static-duty-score 12 \
    --deployable-static-duty-feedback 2.5 \
    --compare-deployable-static \
    --energy-account \
    --energy-capacity 180 \
    --initial-energy 180 \
    --harvest-per-step 0.82 \
    --reserve-energy 20 \
    --require-diverse-dynamic \
    --min-dynamic-margin 0.005 \
    --min-mid-duty-sensors 5 \
    --max-always-on-sensors 1 \
    --max-always-off-sensors 2 \
    --min-switches-per-step 0.003 \
    --max-switches-per-step 0.08 \
    --seed "$seed" \
    --force \
    --force-truth
done
