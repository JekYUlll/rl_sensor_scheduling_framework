#!/usr/bin/env bash
set -euo pipefail

# V293 exposes the already-generated legal nowcast and forecast-quality
# columns to the context encoder. Reward, arbitrary feasible subsets, fixed
# effective costs, scene generator, and ordinary PPO training remain matched
# to V279; no event labels or baseline actions are added.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$#" -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(6801 6802); fi
GPU_OFFSET="${GPU_OFFSET:-0}"

run_one() {
  local seed="$1" gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_PREFIX="v293_forecast_quality_context_pdppo_sixch_dev"
    export LOG_PREFIX="v293_forecast_quality_context_pdppo_sixch"
    export TOTAL_TIMESTEPS=50000
    export REWARD_PROXY_MODE=forecast EVENT_START_PROB=1.0
    export DECISION_ONLY_POLICY_UPDATES=0 CHECKPOINT_SELECTION_INTERVAL_UPDATES=5
    export CHECKPOINT_SELECTION_SCORE=oracle_loss_mean CHECKPOINT_REQUIRE_VALID_BEHAVIOR=1
    export BC_PRETRAIN_STEPS=0 BC_PRETRAIN_EPOCHS=0 BC_PRETRAIN_LOSS_COEF=0
    export FORECAST_VALUE_AUX_COEF=0 FORECAST_VALUE_RANKING_COEF=0
    export CANDIDATE_INTERACTION_SCORE=0 FACTORIZED_ACTION_POLICY=0
    export AGENT_CONTEXT_COLUMNS="agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c agent_context_nowcast_solar_radiation_wm2 agent_context_quality_forecast_met_station_core agent_context_quality_forecast_radiometer_basic agent_context_quality_forecast_shielded_thermo_hygro agent_context_quality_forecast_surface_temp_ir agent_context_quality_forecast_laser_disdrometer agent_context_quality_forecast_fc4_flux"
    export CONTEXT_FEATURE_DIM=30
    export CONTROL_SOURCE_RUN_DIR="reports/v279_eventbalanced_forecast_checkpoint_pdppo_sixch_dev_seed${seed}_b1p75_20260822"
    bash scripts/run_v267_block_gain_pdppo_sixch_dev_20260830.sh "$seed"
  ) >"logs/v293_forecast_quality_context_pdppo_sixch_seed${seed}.log" 2>&1
}

mkdir -p logs
pids=()
for i in "${!SEEDS[@]}"; do run_one "${SEEDS[$i]}" "$((i + GPU_OFFSET))" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
