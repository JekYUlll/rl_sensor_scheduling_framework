#!/usr/bin/env bash
set -euo pipefail

# V355 isolates reward-scale alignment on the V352 within-event cycling scene.
# It retains the V353/V354 policy, action geometry, online information boundary,
# forecast evaluator, and hard execution constraints, changing only the
# environment reward normalization from subtype-normalized to raw forecast
# loss. No comparator signal or test label is used.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCENE_SEEDS_OVERRIDE="${SCENE_SEEDS_OVERRIDE:-6891 6892}" \
POLICY_SEEDS_OVERRIDE="${POLICY_SEEDS_OVERRIDE:-7421 7422}" \
CONTROL_SOURCE_RUN_DIR_PREFIX="${CONTROL_SOURCE_RUN_DIR_PREFIX:-reports/v352_cycling_scene_control}" \
EVENT_SUBTYPE_ASSIGNMENT_OVERRIDE=cycling \
EVENT_SUBTYPE_CYCLE_STEPS_OVERRIDE=12 \
REWARD_LOSS_NORMALIZATION_OVERRIDE=none \
RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v355_plain_reward_cycling_pdppo_dev}" \
LOG_PREFIX_OVERRIDE="${LOG_PREFIX_OVERRIDE:-v355_plain_reward_cycling_pdppo}" \
bash scripts/run_v346_onpolicy_forecast_aux_direct_mask_20260901.sh
