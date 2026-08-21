#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROFILE_NAME="${PROFILE_NAME:-particle_heavy_flux_v7}"
BUDGET="${BUDGET:-1.03}"
BUDGET_TAG="${BUDGET_TAG:-1p03}"

export PROFILE_NAME
export BUDGET
export BUDGET_TAG
export SENSOR_CFG="${SENSOR_CFG:-configs/sensors/windblown_sensors_physical_event_v26_calm_selective.yaml}"
export SOURCE_DIR="${SOURCE_DIR:-reports/v31_static_break_v26_calm_selective_low_budget_${PROFILE_NAME}_b${BUDGET_TAG}_zero_ppo_source_seed45_h082_20260620}"
export REPLAY_DIR="${REPLAY_DIR:-reports/v31_static_break_v26_calm_selective_low_budget_${PROFILE_NAME}_b${BUDGET_TAG}_split_replay_gate_seed45_h082_20260620}"

exec "$SCRIPT_DIR/run_pdppo_static_break_v25_v24_low_budget_split_replay_gate_seed45_h082_20260620.sh"
