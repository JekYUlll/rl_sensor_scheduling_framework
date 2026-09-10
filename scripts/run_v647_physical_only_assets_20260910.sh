#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
TRUTH_ROOT="${V647_TRUTH_ROOT:-reports/v645_empirical_cold_wind_truth_20260910}"
RESOURCE_ROOT="${V647_RESOURCE_ROOT:-reports/v614_empirical_cold_availability_truth_20260910}"
OUT_ROOT="${V647_ASSET_ROOT:-reports/v647_physical_only_assets_20260910}"
SENSOR_CFG="configs/sensors/windblown_sensors_entity_effective_cost_v1.yaml"

mkdir -p "$OUT_ROOT"
for seed in 7401 7402 7403 7404; do
  truth="$TRUTH_ROOT/truth/truth_seed${seed}_cold_wind.csv"
  trace="$RESOURCE_ROOT/resource/resource_seed${seed}_heater.csv"
  out="$OUT_ROOT/seed${seed}"
  "$PYTHON_BIN" scripts/25_v2_train_custom_ppo.py \
    --truth-csv "$truth" --sensor-cfg "$SENSOR_CFG" --out-dir "$out" \
    --seed "$seed" --policy-seed "$seed" --exclude-subtype-latents-from-state \
    --sensor-quality-columns \
      agent_context_quality_met_station_core agent_context_quality_radiometer_basic \
      agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer \
      agent_context_quality_fc4_flux agent_context_quality_cr1000xe_backbone \
    --sensor-quality-max-noise-multiplier 3.0 --sensor-quality-availability-floor 0.0 \
    --dynamic-resource-trace "$trace" --dynamic-resource-budget-w 55.0 \
    --include-dynamic-resource-state --required-sensors cr1000xe_backbone \
    --per-step-budget 10.0 --startup-peak-budget 10.0 --disable-coverage-groups \
    --min-dwell-steps 6 --horizon 8 --lookback 20 --truth-steps 90000 \
    --oracle-start-idx 0 --oracle-end-idx 31500 \
    --normalization-start-idx 31500 --normalization-end-idx 76500 \
    --train-start-min 31500 --train-start-max 75963 \
    --oracle-type tcn --oracle-inference-device cpu --oracle-rollout-steps 1200 \
    --oracle-rollouts-per-policy 4 --candidate-prior-steps 1024 \
    --candidate-prior-rollouts 4 \
    --candidate-prior-start-indices 76589 77858 79988 82985 86293 86811 88490 89496 \
    --static-selection-start-indices 76589 77858 79988 82985 86293 86811 88490 89496 \
    --eval-start-indices 76589 77858 79988 82985 86293 86811 88490 89496 \
    --eval-steps 256 --eval-rollouts 6 --event-start-prob 0.67 \
    --evaluation-policy-mode deterministic --total-timesteps 0 --prepare-assets-only
done
