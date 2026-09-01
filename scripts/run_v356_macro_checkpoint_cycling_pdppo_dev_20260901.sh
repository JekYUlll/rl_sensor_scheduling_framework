#!/usr/bin/env bash
set -euo pipefail

# V356 isolates checkpoint selection. It restores the V353 normalized forecast
# reward and changes only the validation selection endpoint to the co-primary
# macro score; no final-partition feedback or privileged test labels are used.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCENE_SEEDS_OVERRIDE="${SCENE_SEEDS_OVERRIDE:-6891 6892}" \
POLICY_SEEDS_OVERRIDE="${POLICY_SEEDS_OVERRIDE:-7431 7432}" \
CONTROL_SOURCE_RUN_DIR_PREFIX="${CONTROL_SOURCE_RUN_DIR_PREFIX:-reports/v352_cycling_scene_control}" \
EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE=cycling \
EVENT_SUBTYPE_CYCLE_STEPS_OVERRIDE=12 \
REWARD_LOSS_NORMALIZATION_OVERRIDE=staticnorm_subtype \
CHECKPOINT_SELECTION_SCORE_OVERRIDE=oracle_loss_macro_subtype_event_staticnorm \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v356_macro_checkpoint_cycling_pdppo_dev}" \
LOG_PREFIX_OVERRIDE="${LOG_PREFIX_OVERRIDE:-v356_macro_checkpoint_cycling_pdppo}" \
bash scripts/run_v346_onpolicy_forecast_aux_direct_mask_20260901.sh
