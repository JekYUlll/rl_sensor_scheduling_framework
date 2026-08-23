#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Development-only online-identifiability screen. Costs, action geometry, truth
# dynamics, and evaluator settings match V103; only the synthetic online warning
# confidence gains a noisy forecast-intensity component.
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v113_frequency_cost_intensity_context_gate_dev}" \
CONTEXT_OUT_OVERRIDE="${CONTEXT_OUT_OVERRIDE:-reports/aggregate/v113_frequency_cost_intensity_context_gate_20260823}" \
EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH_OVERRIDE="${EVENT_SUBTYPE_CONTEXT_LATENT_STRENGTH_OVERRIDE:-0.75}" \
bash scripts/run_v103_frequency_cost_scene_gate_20260823.sh "${1:-all}"
