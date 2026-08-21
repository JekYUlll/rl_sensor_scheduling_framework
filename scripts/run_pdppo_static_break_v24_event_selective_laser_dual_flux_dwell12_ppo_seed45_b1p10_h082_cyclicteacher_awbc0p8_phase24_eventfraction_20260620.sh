#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export OUT_DIR="reports/v31_static_break_v24_event_selective_laser_dual_flux_dwell12_ppo_seed45_b1p10_h082_cyclicteacher_awbc0p8_phase24_eventfraction_20260620"
export SUMMARY_NAME="v24_eventlaser_dualflux_cyclicteacher_awbc0p8_phase24_seed45_h082_eventfraction_summary.csv"
export AWBC_COEF="0.80"
export TARGET_WEIGHTS="0.03 0.03 0.10 0.01 0.01 0.0 22.0 16.0 16.0"
export INCLUDE_AGENT_CYCLE_PHASE="1"
export AGENT_CYCLE_PERIOD_STEPS="24"
export AGENT_CYCLE_DWELL_STEPS="12"

exec "$SCRIPT_DIR/run_pdppo_static_break_v24_event_selective_laser_dual_flux_dwell12_ppo_seed45_b1p10_h082_cyclicteacher_awbc0p8_eventfraction_20260620.sh"
