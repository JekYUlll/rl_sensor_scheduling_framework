#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
TRUTH_ROOT="${V524_TRUTH_ROOT:-reports/v521_independent_target_innovation_truth_20260908}"
OUT_ROOT="${V524_OUT_ROOT:-reports/v524_entity_effective_cost_assets_20260908}"
SENSOR_CFG="configs/sensors/windblown_sensors_entity_effective_cost_v1.yaml"

mkdir -p "$OUT_ROOT"
for seed in 7177 7178 7179 7180; do
  DERIVED_TRUTH="$OUT_ROOT/truth_seed${seed}_entity_effective.csv"
  "$PYTHON_BIN" scripts/128_prepare_entity_effective_truth.py \
    --input "$TRUTH_ROOT/truth_seed${seed}.csv" --output "$DERIVED_TRUTH"
  "$PYTHON_BIN" scripts/58_v31_split_protocol_run.py \
    --out-dir "$OUT_ROOT/seed${seed}_b2p15" \
    --truth-csv "$DERIVED_TRUTH" \
    --seed "$seed" --policy-seed "$((9290 + seed - 7177))" \
    --sensor-cfg "$SENSOR_CFG" \
    --required-sensors cr1000xe_backbone \
    --disable-coverage-groups \
    --budget 2.15 --startup-peak-budget 2.60 \
    --truth-steps 90000 --lookback 20 --forecast-horizon 24 \
    --split-ratios 0.35 0.50 0.075 0.075 --event-coverage 0.45 \
    --min-duration 20 --max-duration 64 --min-gap 12 --lead-steps 8 \
    --nowcast-lead-steps 6 --channel-quality-enabled \
    --channel-quality-mode condition_dependent_crossover_robust \
    --channel-quality-sensor-ids met_station_core radiometer_basic surface_temp_ir laser_disdrometer fc4_flux \
    --sensor-quality-columns agent_context_quality_met_station_core agent_context_quality_radiometer_basic agent_context_quality_surface_temp_ir agent_context_quality_laser_disdrometer agent_context_quality_fc4_flux agent_context_quality_cr1000xe_backbone \
    --agent-context-columns agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c agent_context_forecast_mode_transport agent_context_forecast_mode_particle agent_context_forecast_mode_thermal \
    --exclude-subtype-latents-from-state --sensor-quality-max-noise-multiplier 3.0 \
    --prepare-assets-only
done
