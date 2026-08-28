#!/usr/bin/env bash
set -euo pipefail

# Development-only admission gate for a five-instrument physical system. The
# fixed effective acquisition loads, one scheduling interval, and arbitrary
# feasible subsets are inherited from V232. Only the weather-observable
# reliability mechanism changes: instruments have crossing physical exposure
# and signal-to-noise profiles under ordinary meteorological conditions.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2601 2602 2603 2604 2605}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v234_physical_crossover_dev}"
export CONTEXT_OUT_OVERRIDE="${CONTEXT_OUT_OVERRIDE:-reports/aggregate/v234_physical_crossover_context_20260828}"
export OUT_ROOT="${OUT_ROOT:-reports/aggregate/v234_physical_crossover_gate_20260828}"
export CHANNEL_QUALITY_ENABLED_OVERRIDE=1
export CHANNEL_QUALITY_MODE_OVERRIDE=condition_dependent_crossover
export CHANNEL_QUALITY_DEGRADED_COVERAGE_OVERRIDE=0.0
export CHANNEL_QUALITY_DEGRADED_VALUE_OVERRIDE=0.10
export CHANNEL_QUALITY_REPORT_NOISE_STD_OVERRIDE=0.02
export SENSOR_QUALITY_MAX_NOISE_MULTIPLIER_OVERRIDE=6.0
export SENSOR_QUALITY_AVAILABILITY_FLOOR_OVERRIDE=0.10

exec bash scripts/run_v232_physical_weather_quality_scene_gate_20260828.sh "${1:-all}"
