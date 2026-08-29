#!/usr/bin/env bash
set -euo pipefail

# Development-only admission gate. Five independently powered physical groups
# retain arbitrary feasible subsets and fixed effective loads. A noisy health
# diagnostic is generated from weather exposure; no event label, rule action,
# or sampling-frequency action is exposed to the scheduler.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2501 2502 2503 2504 2505}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v232_physical_weather_quality_dev}"
export CONTEXT_OUT_OVERRIDE="${CONTEXT_OUT_OVERRIDE:-reports/aggregate/v232_physical_weather_quality_context_20260828}"
export OUT_ROOT="${OUT_ROOT:-reports/aggregate/v232_physical_weather_quality_gate_20260828}"
export SENSOR_CFG_OVERRIDE="${SENSOR_CFG_OVERRIDE:-configs/sensors/windblown_sensors_physical_groups_v1.yaml}"
export BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-1.85}"
export STARTUP_BUDGET_OVERRIDE="${STARTUP_BUDGET_OVERRIDE:-2.25}"
export BUDGET_LABEL_OVERRIDE="${BUDGET_LABEL_OVERRIDE:-b1p85}"
export TOTAL_TIMESTEPS_OVERRIDE="${TOTAL_TIMESTEPS_OVERRIDE:-1024}"
export EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE=stratified_duration
export EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY_OVERRIDE=0.0
export CHANNEL_QUALITY_ENABLED_OVERRIDE=1
export CHANNEL_QUALITY_MODE_OVERRIDE="${CHANNEL_QUALITY_MODE_OVERRIDE:-condition_dependent}"
export CHANNEL_QUALITY_DEGRADED_COVERAGE_OVERRIDE=0.0
export CHANNEL_QUALITY_DEGRADED_VALUE_OVERRIDE=0.25
export CHANNEL_QUALITY_REPORT_NOISE_STD_OVERRIDE=0.03
export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER_OVERRIDE=5.0
export SENSOR_QUALITY_AVAILABILITY_FLOOR_OVERRIDE=0.20
export CHANNEL_QUALITY_SENSOR_IDS="gmx500_weather_station lps10_pyranometer si111_surface_ir parsivel2_disdrometer flowcapt_fc4"
export SENSOR_QUALITY_COLUMNS="agent_context_quality_gmx500_weather_station agent_context_quality_lps10_pyranometer agent_context_quality_si111_surface_ir agent_context_quality_parsivel2_disdrometer agent_context_quality_flowcapt_fc4"
export AGENT_CONTEXT_COLUMNS="agent_context_nowcast_wind_speed_ms agent_context_nowcast_relative_humidity agent_context_nowcast_air_temperature_c"
export CONTEXT_FEATURE_DIM_OVERRIDE=8
export INCLUDE_ALERT_CONTEXT_FEATURES=0
export NOWCAST_LEAD_STEPS=8
export NOWCAST_WIND_NOISE_STD=1.4
export NOWCAST_HUMIDITY_NOISE_STD=4.2
export NOWCAST_TEMPERATURE_NOISE_STD=1.0

# The generic scene runner consumes the override name, while the two local
# post-scene phases use a concrete path.
CONTEXT_OUT="$CONTEXT_OUT_OVERRIDE"

phase="${1:-all}"
case "$phase" in
  scene) exec bash scripts/run_v137_generic_physical_scene_gate_20260826.sh scene ;;
  receding) exec bash scripts/run_v137_generic_physical_scene_gate_20260826.sh receding ;;
  context)
    read -r -a seeds <<< "$SEEDS_OVERRIDE"
    mkdir -p "$CONTEXT_OUT" logs
    pids=()
    for idx in "${!seeds[@]}"; do
      seed="${seeds[$idx]}"
      (
        export CUDA_VISIBLE_DEVICES="$idx"
        "$HOME/.conda/envs/darts/bin/python" scripts/81_v31_framework_baseline_supplements.py \
          --run-glob "reports/${RUN_PREFIX_OVERRIDE}_seed${seed}_${BUDGET_LABEL_OVERRIDE}_20260822" \
          --seeds "$seed" --out-root "$CONTEXT_OUT" --router-eval-dir . \
          --replay-dir __none__ --oracle-device cuda --policies quality_only \
          --quality-penalties 1.0 --no-aggregate
      ) >"logs/v232_quality_context_seed${seed}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    ;;
  collect)
    read -r -a seeds <<< "$SEEDS_OVERRIDE"
    exec "$HOME/.conda/envs/darts/bin/python" scripts/107_v32_collect_physical_quality_gate.py \
      --prefix "$RUN_PREFIX_OVERRIDE" --budget-label "$BUDGET_LABEL_OVERRIDE" \
      --context-root "$CONTEXT_OUT" --out-dir "$OUT_ROOT" --seeds "${seeds[@]}" \
      --quality-policy quality_only_calibrated_p1p0
    ;;
  all)
    bash "$0" scene
    bash "$0" context
    bash "$0" receding
    exec bash "$0" collect
    ;;
  *) printf 'unknown phase: %s\n' "$phase" >&2; exit 2 ;;
esac
