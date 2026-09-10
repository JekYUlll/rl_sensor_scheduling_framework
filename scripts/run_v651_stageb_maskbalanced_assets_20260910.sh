#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
TRUTH_ROOT="${V651_TRUTH_ROOT:-reports/v535_persistent_transport_truth_20260909}"
RESOURCE_ROOT="${V651_RESOURCE_ROOT:-reports/v535_persistent_transport_resource_20260909}"
OUT_ROOT="${V651_OUT_ROOT:-reports/v651_stageb_maskbalanced_assets_20260910}"
SENSOR_CFG="configs/sensors/windblown_sensors_physical_power_v1.yaml"

mkdir -p "$OUT_ROOT"
for seed in 7231 7232 7233 7234; do
  truth="$OUT_ROOT/truth_seed${seed}_entity_effective.csv"
  "$PYTHON_BIN" scripts/128_prepare_entity_effective_truth.py \
    --input "$TRUTH_ROOT/truth_seed${seed}_persistent_transport.csv" \
    --output "$truth"
  "$PYTHON_BIN" scripts/25_v2_train_custom_ppo.py \
    --truth-csv "$truth" --sensor-cfg "$SENSOR_CFG" \
    --out-dir "$OUT_ROOT/seed${seed}" --seed "$seed" --policy-seed "$seed" \
    --exclude-subtype-latents-from-state \
    --sensor-quality-columns \
      agent_context_quality_met_station_core agent_context_quality_radiometer_basic \
      agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer \
      agent_context_quality_fc4_flux agent_context_quality_cr1000xe_backbone \
    --sensor-quality-max-noise-multiplier 3.0 --sensor-quality-availability-floor 0.2 \
    --dynamic-resource-trace "$RESOURCE_ROOT/resource_seed${seed}.csv" \
    --dynamic-resource-budget-w 20.0 --include-dynamic-resource-state \
    --required-sensors cr1000xe_backbone --per-step-budget 20.0 \
    --startup-peak-budget 25.0 --disable-coverage-groups --min-dwell-steps 6 \
    --horizon 8 --lookback 20 --truth-steps 90000 \
    --oracle-start-idx 0 --oracle-end-idx 31500 \
    --normalization-start-idx 31500 --normalization-end-idx 76500 \
    --train-start-min 31500 --train-start-max 75963 \
    --oracle-type tcn --oracle-inference-device cpu --oracle-rollout-steps 1200 \
    --oracle-rollouts-per-policy 4 --candidate-prior-steps 1024 \
    --candidate-prior-rollouts 4 --candidate-prior-start-indices 77452 78907 80344 81789 \
    --static-selection-start-indices 77452 78907 80344 81789 \
    --eval-start-indices 77452 78907 80344 81789 --eval-steps 256 \
    --eval-rollouts 6 --event-start-prob 0.67 \
    --evaluation-policy-mode deterministic --total-timesteps 0 --prepare-assets-only \
    --oracle-loss-clip 100 --oracle-candidate-mask-repeat 1
done
