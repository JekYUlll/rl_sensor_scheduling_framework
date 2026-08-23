#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Information-only development gate. Replace the mixed binary/intensity warning
# score with the noisy lead intensity forecast while keeping truth dynamics,
# costs, action geometry, evaluator, and partitions matched to V113.
RUN_PREFIX_OVERRIDE=v120_full_intensity_context_gate_dev \
CONTEXT_OUT_OVERRIDE=reports/aggregate/v120_full_intensity_context_gate_20260823 \
EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH_OVERRIDE=1.0 \
bash scripts/run_v103_frequency_cost_scene_gate_20260823.sh "${1:-all}"
