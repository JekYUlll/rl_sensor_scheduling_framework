#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Matched V107 observability probe with a low-temperature continuous
# forecast-value target instead of hard argmin action labels.
POLICY_SEED_OFFSET="${POLICY_SEED_OFFSET:-6000}" \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v108_frequency_cost_soft_value_probe}" \
BC_PRETRAIN_TARGET_MODE_OVERRIDE=soft_forecast_value \
BC_SOFT_TEMPERATURE_OVERRIDE="${BC_SOFT_TEMPERATURE_OVERRIDE:-0.05}" \
bash scripts/run_v107_frequency_cost_observability_probe_20260823.sh
