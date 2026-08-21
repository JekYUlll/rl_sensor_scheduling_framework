#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROFILE_NAME="${PROFILE_NAME:-particle_heavy_flux_v7}"
BUDGET="${BUDGET:-1.08}"
BUDGET_TAG="${BUDGET_TAG:-1p08}"

export PROFILE_NAME
export BUDGET
export BUDGET_TAG
export SENSOR_CFG="${SENSOR_CFG:-configs/sensors/windblown_sensors_physical_event_v26_calm_selective.yaml}"
export OUT_DIR="${OUT_DIR:-reports/v31_static_break_v26_calm_selective_low_budget_${PROFILE_NAME}_b${BUDGET_TAG}_learned_ppo_seed45_h082_20260620}"
export SUMMARY_NAME="${SUMMARY_NAME:-v26_calm_selective_lowbudget_${PROFILE_NAME}_b${BUDGET_TAG}_learned_ppo_seed45_h082_summary.csv}"

exec "$SCRIPT_DIR/run_pdppo_static_break_v25_v24_low_budget_learned_ppo_seed45_h082_20260620.sh"
