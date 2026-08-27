#!/usr/bin/env bash
set -euo pipefail

# Development-only screen for the fixed-load calibrated physical channels.
# The action surface remains arbitrary feasible subsets: the budget removes
# only combinations whose simultaneous channel loads exceed the fixed limit.
cd "$(dirname "$0")/.."

COMMON_RANDOM_NUMBERS=1 \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2201 2202 2203 2204 2205}" \
RUN_PREFIX_OVERRIDE=v212_coverage_balanced_scene_dev \
CONTEXT_OUT_OVERRIDE=reports/aggregate/v212_coverage_balanced_context_gate_20260828 \
SENSOR_CFG_OVERRIDE=configs/sensors/windblown_sensors_flexible_subset_v7_coverage_balanced.yaml \
BUDGET_OVERRIDE=1.85 \
STARTUP_BUDGET_OVERRIDE=2.25 \
BUDGET_LABEL_OVERRIDE=b1p85 \
EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY_OVERRIDE=0.0 \
CHANNEL_QUALITY_ENABLED_OVERRIDE=1 \
CHANNEL_QUALITY_MODE_OVERRIDE=independent \
CHANNEL_QUALITY_DEGRADED_COVERAGE_OVERRIDE=0.25 \
CHANNEL_QUALITY_MIN_DURATION_STEPS_OVERRIDE=24 \
CHANNEL_QUALITY_MAX_DURATION_STEPS_OVERRIDE=64 \
CHANNEL_QUALITY_MIN_GAP_STEPS_OVERRIDE=16 \
CHANNEL_QUALITY_DEGRADED_VALUE_OVERRIDE=0.2 \
CHANNEL_QUALITY_TRANSITION_STEPS_OVERRIDE=8 \
CHANNEL_QUALITY_REPORT_NOISE_STD_OVERRIDE=0.02 \
SENSOR_QUALITY_MAX_NOISE_MULTIPLIER_OVERRIDE=7.0 \
SENSOR_QUALITY_AVAILABILITY_FLOOR_OVERRIDE=0.05 \
bash scripts/run_v137_generic_physical_scene_gate_20260826.sh "${1:-all}"
