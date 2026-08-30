#!/usr/bin/env bash
set -euo pipefail

# V244 enables the existing action-conditioned quality representation.  The
# five quality forecasts are placed first because the model interface defines
# them as the leading channel features; the four weather nowcasts are retained
# as context for the learned channel utility encoder.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RUN_PREFIX="v244_quality_action_context_pdppo_dev"
export LOG_PREFIX="v244_quality_action_context_pdppo"
export AGENT_CONTEXT_COLUMNS="agent_context_quality_forecast_gmx500_weather_station agent_context_quality_forecast_lps10_pyranometer agent_context_quality_forecast_si111_surface_ir agent_context_quality_forecast_parsivel2_disdrometer agent_context_quality_forecast_flowcapt_fc4 agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c agent_context_nowcast_solar_radiation_wm2"
export QUALITY_CONTEXT_ACTION_SCORE=1
exec bash scripts/run_v243_full_pdppo_no_awbc_balanced_quality_20260830.sh "${@:-3501 3502 3503 3504 3505}"
