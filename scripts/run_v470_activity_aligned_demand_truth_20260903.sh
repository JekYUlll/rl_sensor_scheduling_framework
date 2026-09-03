#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUN_PREFIX_OVERRIDE=v470_activity_aligned_demand_truth
export LOG_PREFIX_OVERRIDE=v470_activity_aligned_demand_truth
export SCENE_SEEDS_OVERRIDE="7091 7092"
export POLICY_SEEDS_OVERRIDE="9121 9122"
export USE_CONTROL_SOURCE_OVERRIDE=0 PREPARE_TRUTH_ONLY=1
export EVENT_COVERAGE_OVERRIDE=0.45 EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE=random
export CHANNEL_QUALITY_MODE_OVERRIDE=condition_dependent_crossover_robust
export CONTINUOUS_OPERATING_STATE=0 EXPOSURE_RECOVERY_STATE=0
export BALANCED_EXPOSURE_RECOVERY_STATE=0 DECOUPLED_EXPOSURE_RECOVERY_STATE=0
export THREE_FACTOR_EXPOSURE_STATE=0
export FORECAST_VALUE_STATE=1 FORECAST_VALUE_STATIONARY_LOCAL_STATE=0
export FORECAST_VALUE_RESIDENCE_LOCAL_STATE=1
export FORECAST_VALUE_HORIZON_PERSISTENT_LATENT=1
export FORECAST_VALUE_SPECIALIST_RESILIENT_QUALITY=1
export FORECAST_VALUE_ACTIVITY_ALIGNED_TRANSPORT_DEMAND=1
export AGENT_CONTEXT_COLUMNS_OVERRIDE="agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c agent_context_nowcast_solar_radiation_wm2 agent_context_forecast_flux_demand agent_context_forecast_particle_demand agent_context_forecast_thermal_demand"
export INCLUDE_ALERT_CONTEXT_FEATURES_OVERRIDE=0 EVENT_AWARE_CRITIC_OVERRIDE=0
export SUBTYPE_LOSS_WEIGHTING_OVERRIDE=0 SUBTYPE_AUX_COEF_OVERRIDE=0
export REWARD_LOSS_NORMALIZATION_OVERRIDE=none

status="logs/v470_activity_aligned_demand_truth_20260903.exit"
rm -f "$status"
bash scripts/run_v361_multiscene_cycling_pdppo_20260901.sh
rc=$?
printf '%s\n' "$rc" >"$status"
exit "$rc"
