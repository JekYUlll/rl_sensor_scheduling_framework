#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
SOURCE_ROOT="${V680_SOURCE_ROOT:-reports/v675_split_affinity_quality_assets_b2p15_20260911}"
TRUTH_ROOT="${V680_TRUTH_ROOT:-reports/v527_observable_target_truth_20260909_r2}"
RESOURCE_ROOT="${V680_RESOURCE_ROOT:-reports/v672_shared_driver_resource_screen_20260911}"
OUT_ROOT="${V680_OUT_ROOT:-reports/v680_observable_target_resource_assets_b2p50_20260911}"

mkdir -p "$OUT_ROOT"
"$PYTHON_BIN" scripts/175_prepare_causal_specialist_assets.py \
  --source-root "$SOURCE_ROOT" \
  --truth-root "$TRUTH_ROOT" \
  --resource-root "$RESOURCE_ROOT" \
  --output-root "$OUT_ROOT" \
  --seeds 7177 7178 7179 7180 \
  --budget 2.50 \
  --objective ordinary \
  --oracle-loss-clip 100 \
  --context-columns \
    agent_context_nowcast_wind_speed_ms \
    agent_context_nowcast_relative_humidity \
    agent_context_nowcast_air_temperature_c \
    agent_context_nowcast_solar_radiation_wm2
