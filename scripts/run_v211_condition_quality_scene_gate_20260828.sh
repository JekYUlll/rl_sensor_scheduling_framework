#!/usr/bin/env bash
set -euo pipefail

# Development-only scene gate for continuous, condition-dependent sensor
# reliability. Physical events are generated first; the quality diagnostics are
# derived afterwards from continuous environmental exposure and include noise.
cd "$(dirname "$0")/.."

COMMON_RANDOM_NUMBERS=1 \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2101 2102 2103 2104 2105}" \
RUN_PREFIX_OVERRIDE=v211_condition_quality_scene_dev \
CONTEXT_OUT_OVERRIDE=reports/aggregate/v211_condition_quality_context_gate_20260828 \
EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY_OVERRIDE=0.0 \
CHANNEL_QUALITY_ENABLED_OVERRIDE=1 \
CHANNEL_QUALITY_MODE_OVERRIDE=condition_dependent \
CHANNEL_QUALITY_DEGRADED_COVERAGE_OVERRIDE=0.0 \
CHANNEL_QUALITY_DEGRADED_VALUE_OVERRIDE=0.35 \
CHANNEL_QUALITY_REPORT_NOISE_STD_OVERRIDE=0.03 \
SENSOR_QUALITY_MAX_NOISE_MULTIPLIER_OVERRIDE=5.0 \
SENSOR_QUALITY_AVAILABILITY_FLOOR_OVERRIDE=0.20 \
bash scripts/run_v137_generic_physical_scene_gate_20260826.sh "${1:-all}"
