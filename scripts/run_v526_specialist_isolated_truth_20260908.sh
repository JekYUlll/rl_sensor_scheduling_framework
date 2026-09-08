#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
OUT="${V526_OUT_ROOT:-reports/v526_specialist_isolated_truth_20260908}"
mkdir -p "$OUT"
for seed in 7181 7182 7183 7184; do
  base="$OUT/base_truth_seed${seed}.csv"
  truth="$OUT/truth_seed${seed}.csv"
  "$PYTHON_BIN" scripts/20_build_public_weather_truth.py \
    --seed "$seed" --steps 90000 --blowing-snow-event-coverage 0.45 \
    --blowing-snow-event-model semi_markov --blowing-snow-min-duration-steps 20 \
    --blowing-snow-max-duration-steps 64 --blowing-snow-min-gap-steps 12 \
    --blowing-snow-lead-steps 8 --event-subtypes-enabled \
    --event-subtype-assignment random --nowcast-lead-steps 6 \
    --channel-quality-enabled --channel-quality-mode condition_dependent_crossover_robust \
    --channel-quality-sensor-ids met_station_core radiometer_basic shielded_thermo_hygro surface_temp_ir laser_disdrometer fc4_flux \
    --out "$base" --report-dir "$OUT/dataset_validation_seed${seed}"
  "$PYTHON_BIN" scripts/129_v32_build_specialist_isolated_target_truth.py \
    --input "$base" --output "$truth" --seed "$seed"
done
