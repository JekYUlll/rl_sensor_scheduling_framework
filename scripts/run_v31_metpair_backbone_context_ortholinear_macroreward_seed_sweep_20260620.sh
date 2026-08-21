#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_macroreward}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolmr}"
export DATE_TAG="${DATE_TAG:-20260620}"

# Start from the ortholinear strong-teacher branch, then align the RL reward
# with event-subtype macro robustness. This changes what PPO optimizes; the
# replay/static gate is still evaluated independently after training.
export EVENT_COVERAGE="${EVENT_COVERAGE:-0.68}"
export SUBTYPE_LATENT_ALPHA="${SUBTYPE_LATENT_ALPHA:-0.45}"
export PARTICLE_LATENT_DIAMETER_SCALE_MM="${PARTICLE_LATENT_DIAMETER_SCALE_MM:-0.30}"
export PARTICLE_LATENT_VELOCITY_SCALE_MS="${PARTICLE_LATENT_VELOCITY_SCALE_MS:-4.5}"
export FLUX_MULTIPLIER="${FLUX_MULTIPLIER:-4.0}"
export FLUX_LATENT_SIGMA="${FLUX_LATENT_SIGMA:-0.0}"
export FLUX_LATENT_LINEAR_SCALE="${FLUX_LATENT_LINEAR_SCALE:-0.0025}"
export FLUX_LATENT_LINEAR_OFFSET="${FLUX_LATENT_LINEAR_OFFSET:-1.5}"
export FLUX_LATENT_LINEAR_CLIP="${FLUX_LATENT_LINEAR_CLIP:-3.5}"
export THERMAL_LATENT_SURFACE_SCALE_C="${THERMAL_LATENT_SURFACE_SCALE_C:-3.0}"
export THERMAL_SURFACE_DROP_C="${THERMAL_SURFACE_DROP_C:-3.0}"
export THERMAL_AIR_TEMP_DROP_C="${THERMAL_AIR_TEMP_DROP_C:-1.0}"

export ORACLE_SUBTYPE_TEACHER_REPEAT="${ORACLE_SUBTYPE_TEACHER_REPEAT:-24}"
export ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS="${ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS:-10}"
export AWBC_COEF="${AWBC_COEF:-2.0}"
export AWBC_LABEL_STRIDE="${AWBC_LABEL_STRIDE:-1}"
export AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS="${AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS:-10}"
export AWBC_TEACHER_DWELL_STEPS="${AWBC_TEACHER_DWELL_STEPS:-6}"
export BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-6.0}"
export BC_PRETRAIN_STEPS="${BC_PRETRAIN_STEPS:-8000}"
export BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS:-16}"
export BC_PRETRAIN_BATCH_SIZE="${BC_PRETRAIN_BATCH_SIZE:-256}"
export SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-3.0}"
export SUBTYPE_AUX_LOOKAHEAD_STEPS="${SUBTYPE_AUX_LOOKAHEAD_STEPS:-10}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-180000}"
export N_STEPS="${N_STEPS:-1024}"
export N_EPOCHS="${N_EPOCHS:-8}"
export ENT_COEF="${ENT_COEF:-0.003}"
export MIN_DWELL_STEPS="${MIN_DWELL_STEPS:-6}"
export LAMBDA_SWITCH="${LAMBDA_SWITCH:-0.001}"

# Macro-reward alignment: flux was the learned macro bottleneck on seed41;
# seed42 already learned macro-positive but replay macro was close negative.
export EVENT_SUBTYPE_PARTICLE_REWARD_MULTIPLIER="${EVENT_SUBTYPE_PARTICLE_REWARD_MULTIPLIER:-1.05}"
export EVENT_SUBTYPE_FLUX_REWARD_MULTIPLIER="${EVENT_SUBTYPE_FLUX_REWARD_MULTIPLIER:-1.35}"
export EVENT_SUBTYPE_THERMAL_REWARD_MULTIPLIER="${EVENT_SUBTYPE_THERMAL_REWARD_MULTIPLIER:-1.15}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(41)
fi

bash scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh "${SEEDS[@]}"

RUNS=()
for seed in "${SEEDS[@]}"; do
  RUNS+=("reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${DATE_TAG}")
done

"${PY:-python}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --out-dir "reports/aggregate/metpair_backbone_context_ortholinear_macroreward_${#SEEDS[@]}seed_macro_${DATE_TAG}"

"${PY:-python}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --router-eval-dir . \
  --out-dir "reports/aggregate/metpair_backbone_context_ortholinear_macroreward_${#SEEDS[@]}seed_raw_macro_${DATE_TAG}"
