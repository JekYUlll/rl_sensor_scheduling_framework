#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Frozen V125 expansion. Seed1303 is reused from the completed pilot.
SEEDS_OVERRIDE="1301 1302 1304 1305" \
GPU_IDS="0 1 2 3" \
RUN_PREFIX_OVERRIDE=v126_dense_onpolicy_forecast_value \
bash scripts/run_v125_dense_onpolicy_forecast_value_pilot_20260823.sh
