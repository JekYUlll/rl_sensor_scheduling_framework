#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
SOURCE_ROOT="${V641_SOURCE_ROOT:-reports/v528_observable_target_dynamic_resource_assets_20260909_resourceaudit}"
TRUTH_ROOT="${V641_TRUTH_ROOT:-reports/v640_radiometer_signal_truth_20260910}"
OUT_ROOT="${V641_ASSET_ROOT:-reports/v641_radiometer_signal_assets_b2p15_20260910}"
cd "$ROOT"
mkdir -p "$OUT_ROOT"
"$PYTHON_BIN" scripts/175_prepare_causal_specialist_assets.py \
  --source-root "$SOURCE_ROOT" --truth-root "$TRUTH_ROOT" \
  --resource-root "$TRUTH_ROOT" --output-root "$OUT_ROOT" \
  --seeds 7177 7178 7179 7180 --budget 2.15 \
  --objective ordinary --oracle-loss-clip 1000000000 \
  --context-columns \
    agent_context_operating_factor_transport \
    agent_context_operating_factor_particle \
    agent_context_operating_factor_thermal \
    agent_context_nowcast_wind_speed_ms \
    agent_context_nowcast_air_temperature_c \
    agent_context_nowcast_relative_humidity \
    agent_context_nowcast_solar_radiation_wm2
