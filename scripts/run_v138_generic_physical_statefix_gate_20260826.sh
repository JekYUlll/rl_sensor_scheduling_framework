#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_PREFIX_OVERRIDE=v138_generic_physical_statefix_gate_dev \
CONTEXT_OUT_OVERRIDE=reports/aggregate/v138_generic_physical_context_gate_20260826 \
bash scripts/run_v137_generic_physical_scene_gate_20260826.sh "${1:-all}"
