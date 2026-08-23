#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Cohesive context-aware PD-PPO: online context MoE, training-only subtype
# representation auxiliary, all-action forecast-value initialization, and the
# unchanged forecast-loss PPO objective. Test execution receives no labels.
BC_PRETRAIN_TARGET_MODE_OVERRIDE=forecast_value_regression \
CONTEXT_FUSION_MODE_OVERRIDE=subtype_moe \
SUBTYPE_AUX_COEF_OVERRIDE=0.3 \
CHECKPOINT_SELECTION_SCORE_OVERRIDE=max_static_ratio \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v118_ca_pdppo_jointselect}" \
bash scripts/run_v116_intensity_temporal_forecastbc_ckptppo_20260823.sh
