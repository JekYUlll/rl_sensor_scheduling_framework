#!/usr/bin/env bash
set -euo pipefail

# Feature-aligned PD-PPO screen for the V227 physical instrument grouping. The
# policy receives only a noisy eight-step meteorological forecast; it does not
# receive synthetic event labels, alert features, heuristic actions, or a
# forecast-value/behaviour-cloning target.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2481 2482 2483 2484 2485}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v229_physical_groups_nowcast_ppo_dev}"
export TOTAL_TIMESTEPS_OVERRIDE="${TOTAL_TIMESTEPS_OVERRIDE:-50000}"
export AGENT_CONTEXT_COLUMNS="agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c"
export CONTEXT_FEATURE_DIM_OVERRIDE=3
export NOWCAST_LEAD_STEPS=8
export NOWCAST_WIND_NOISE_STD=1.4
export NOWCAST_HUMIDITY_NOISE_STD=4.2
export NOWCAST_TEMPERATURE_NOISE_STD=1.0

exec bash scripts/run_v227_physical_groups_scene_gate_20260828.sh scene
