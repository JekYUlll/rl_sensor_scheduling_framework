#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolbo}"
export DATE_TAG="${DATE_TAG:-20260620}"

# This branch changes the experimental contract from step-weighted loss to a
# regime-balanced event-subtype objective. Static selection, primary metrics
# sorting, replay macro gate, and claim collection all use the same macro
# subtype target instead of relying on posthoc reinterpretation.
export STATIC_SELECTION_SCORE="${STATIC_SELECTION_SCORE:-oracle_loss_macro_subtype_event_staticnorm}"
export METRICS_SORT_SCORE="${METRICS_SORT_SCORE:-oracle_loss_macro_subtype_event_staticnorm}"
export MACRO_SCORE_COLUMN="${MACRO_SCORE_COLUMN:-oracle_loss_macro_subtype_event_staticnorm}"
export REWARD_LOSS_NORMALIZATION="${REWARD_LOSS_NORMALIZATION:-staticnorm_subtype}"

# Start from the strongest ortholinear teacher configuration.
export EVENT_COVERAGE="${EVENT_COVERAGE:-0.68}"
export SUBTYPE_LATENT_ALPHA="${SUBTYPE_LATENT_ALPHA:-0.50}"
export PARTICLE_LATENT_DIAMETER_SCALE_MM="${PARTICLE_LATENT_DIAMETER_SCALE_MM:-0.36}"
export PARTICLE_LATENT_VELOCITY_SCALE_MS="${PARTICLE_LATENT_VELOCITY_SCALE_MS:-5.2}"
export FLUX_MULTIPLIER="${FLUX_MULTIPLIER:-3.6}"
export FLUX_LATENT_SIGMA="${FLUX_LATENT_SIGMA:-0.0}"
export FLUX_LATENT_LINEAR_SCALE="${FLUX_LATENT_LINEAR_SCALE:-0.0022}"
export FLUX_LATENT_LINEAR_OFFSET="${FLUX_LATENT_LINEAR_OFFSET:-1.5}"
export FLUX_LATENT_LINEAR_CLIP="${FLUX_LATENT_LINEAR_CLIP:-3.3}"
export THERMAL_LATENT_SURFACE_SCALE_C="${THERMAL_LATENT_SURFACE_SCALE_C:-3.8}"
export THERMAL_SURFACE_DROP_C="${THERMAL_SURFACE_DROP_C:-3.8}"
export THERMAL_AIR_TEMP_DROP_C="${THERMAL_AIR_TEMP_DROP_C:-1.4}"

export ORACLE_SUBTYPE_TEACHER_REPEAT="${ORACLE_SUBTYPE_TEACHER_REPEAT:-28}"
export ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS="${ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS:-10}"
export AWBC_COEF="${AWBC_COEF:-2.5}"
export AWBC_LABEL_STRIDE="${AWBC_LABEL_STRIDE:-1}"
export AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS="${AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS:-10}"
export AWBC_TEACHER_DWELL_STEPS="${AWBC_TEACHER_DWELL_STEPS:-6}"
export BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-7.0}"
export BC_PRETRAIN_STEPS="${BC_PRETRAIN_STEPS:-10000}"
export BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS:-18}"
export BC_PRETRAIN_BATCH_SIZE="${BC_PRETRAIN_BATCH_SIZE:-256}"
export SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-3.5}"
export SUBTYPE_AUX_LOOKAHEAD_STEPS="${SUBTYPE_AUX_LOOKAHEAD_STEPS:-10}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
export N_STEPS="${N_STEPS:-1024}"
export N_EPOCHS="${N_EPOCHS:-8}"
export ENT_COEF="${ENT_COEF:-0.0025}"
export MIN_DWELL_STEPS="${MIN_DWELL_STEPS:-6}"
export LAMBDA_SWITCH="${LAMBDA_SWITCH:-0.001}"

# Reward nudges are deliberately modest; the stronger change is the macro
# selection/evaluation contract plus slightly stronger non-flux subtypes.
export EVENT_SUBTYPE_PARTICLE_REWARD_MULTIPLIER="${EVENT_SUBTYPE_PARTICLE_REWARD_MULTIPLIER:-1.30}"
export EVENT_SUBTYPE_FLUX_REWARD_MULTIPLIER="${EVENT_SUBTYPE_FLUX_REWARD_MULTIPLIER:-0.90}"
export EVENT_SUBTYPE_THERMAL_REWARD_MULTIPLIER="${EVENT_SUBTYPE_THERMAL_REWARD_MULTIPLIER:-1.35}"

export TARGET_WEIGHTS="${TARGET_WEIGHTS:-0.04 0.12 0.06 0.0 0.0 0.0 12.0 24.0 24.0}"
export SUBTYPE_PARTICLE_TARGET_WEIGHTS="${SUBTYPE_PARTICLE_TARGET_WEIGHTS:-0.0 0.0 0.08 0.0 0.0 0.0 1.0 60.0 60.0}"
export SUBTYPE_FLUX_TARGET_WEIGHTS="${SUBTYPE_FLUX_TARGET_WEIGHTS:-0.0 0.0 0.05 0.0 0.0 0.0 14.0 3.0 3.0}"
export SUBTYPE_THERMAL_TARGET_WEIGHTS="${SUBTYPE_THERMAL_TARGET_WEIGHTS:-0.12 45.0 0.08 0.0 0.0 0.0 0.2 0.5 0.5}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(41)
fi

bash scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh "${SEEDS[@]}"

if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
  RUNS=()
  for seed in "${SEEDS[@]}"; do
    RUNS+=("reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${DATE_TAG}")
  done

  "${PY:-python}" scripts/72_v31_collect_metpair_strongclaim.py \
    --runs "${RUNS[@]}" \
    --out-dir "reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_${#SEEDS[@]}seed_macro_${DATE_TAG}"

  "${PY:-python}" scripts/72_v31_collect_metpair_strongclaim.py \
    --runs "${RUNS[@]}" \
    --router-eval-dir . \
    --out-dir "reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_${#SEEDS[@]}seed_raw_macro_${DATE_TAG}"
fi
