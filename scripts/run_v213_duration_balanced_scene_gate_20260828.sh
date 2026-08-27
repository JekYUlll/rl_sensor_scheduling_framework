#!/usr/bin/env bash
set -euo pipefail

# Development-only scene gate. The six physical channels, arbitrary feasible
# subsets, costs, budget, and online information are identical to V212. Only
# subtype allocation changes: complete event runs are assigned to balance their
# cumulative occupied duration, not merely their count.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2301 2302 2303 2304 2305}" \
RUN_PREFIX_OVERRIDE=v213_duration_balanced_scene_dev \
CONTEXT_OUT_OVERRIDE=reports/aggregate/v213_duration_balanced_context_gate_20260828 \
SENSOR_CFG_OVERRIDE=configs/sensors/windblown_sensors_flexible_subset_v7_coverage_balanced.yaml \
BUDGET_OVERRIDE=1.85 \
STARTUP_BUDGET_OVERRIDE=2.25 \
BUDGET_LABEL_OVERRIDE=b1p85 \
EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE=stratified_duration \
EVENT_SUBTYPE_PARTICLE_MIN_PARSIVEL_AVAILABILITY_OVERRIDE=0.0 \
CHANNEL_QUALITY_ENABLED_OVERRIDE=1 \
CHANNEL_QUALITY_MODE_OVERRIDE=independent \
bash scripts/run_v137_generic_physical_scene_gate_20260826.sh
