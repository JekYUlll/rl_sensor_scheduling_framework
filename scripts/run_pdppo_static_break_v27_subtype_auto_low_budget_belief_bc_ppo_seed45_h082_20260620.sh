#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PROFILE_NAME="${PROFILE_NAME:-particle_heavy_flux_v7}"
export BUDGET="${BUDGET:-1.05}"
export BUDGET_TAG="${BUDGET_TAG:-1p05}"
export OUT_DIR="${OUT_DIR:-reports/v31_static_break_v27_subtype_auto_low_budget_${PROFILE_NAME}_b${BUDGET_TAG}_belief_bc_ppo_seed45_h082_20260620}"
export SUMMARY_NAME="${SUMMARY_NAME:-v27_subtype_auto_${PROFILE_NAME}_b${BUDGET_TAG}_belief_bc_ppo_seed45_h082_summary.csv}"

export INCLUDE_OBSERVABLE_REGIME_BELIEF="${INCLUDE_OBSERVABLE_REGIME_BELIEF:-1}"
export REGIME_BELIEF_LOOKBACK="${REGIME_BELIEF_LOOKBACK:-8}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000}"
export AWBC_COEF="${AWBC_COEF:-1.00}"
export PRIOR_KL_COEF="${PRIOR_KL_COEF:-0.00}"
export ENT_COEF="${ENT_COEF:-0.0005}"
export BC_PRETRAIN_STEPS="${BC_PRETRAIN_STEPS:-24000}"
export BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS:-10}"
export BC_PRETRAIN_BATCH_SIZE="${BC_PRETRAIN_BATCH_SIZE:-256}"
export BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-1.0}"

exec bash "$SCRIPT_DIR/run_pdppo_static_break_v27_subtype_auto_low_budget_learned_ppo_seed45_h082_20260620.sh"
