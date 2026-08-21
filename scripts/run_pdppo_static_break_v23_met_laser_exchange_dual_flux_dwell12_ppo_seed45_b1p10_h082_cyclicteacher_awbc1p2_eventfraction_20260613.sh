#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export AWBC_COEF=1.20
export OUT_DIR=reports/v31_static_break_v23_met_laser_exchange_dual_flux_dwell12_ppo_seed45_b1p10_h082_cyclicteacher_awbc1p2_eventfraction_20260613
export SUMMARY_NAME=v23_metlaser_dualflux_cyclicteacher_awbc1p2_seed45_h082_eventfraction_summary.csv

exec bash "$SCRIPT_DIR/run_pdppo_static_break_v23_met_laser_exchange_dual_flux_dwell12_ppo_seed45_b1p10_h082_cyclicteacher_awbc0p8_eventfraction_20260613.sh"
