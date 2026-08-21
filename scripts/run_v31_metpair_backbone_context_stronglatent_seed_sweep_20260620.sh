#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_stronglatent}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxsl}"
export DATE_TAG="${DATE_TAG:-20260620}"

export EVENT_COVERAGE="${EVENT_COVERAGE:-0.68}"
export SUBTYPE_LATENT_ALPHA="${SUBTYPE_LATENT_ALPHA:-0.75}"
export PARTICLE_LATENT_DIAMETER_SCALE_MM="${PARTICLE_LATENT_DIAMETER_SCALE_MM:-0.35}"
export PARTICLE_LATENT_VELOCITY_SCALE_MS="${PARTICLE_LATENT_VELOCITY_SCALE_MS:-6.0}"
export FLUX_LATENT_SIGMA="${FLUX_LATENT_SIGMA:-4.0}"
export THERMAL_LATENT_SURFACE_SCALE_C="${THERMAL_LATENT_SURFACE_SCALE_C:-8.0}"
export FLUX_MULTIPLIER="${FLUX_MULTIPLIER:-8.0}"
export THERMAL_SURFACE_DROP_C="${THERMAL_SURFACE_DROP_C:-5.0}"
export THERMAL_AIR_TEMP_DROP_C="${THERMAL_AIR_TEMP_DROP_C:-2.0}"
export ORACLE_SUBTYPE_TEACHER_REPEAT="${ORACLE_SUBTYPE_TEACHER_REPEAT:-16}"
export AWBC_COEF="${AWBC_COEF:-1.0}"
export BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-2.5}"
export SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-2.5}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(41 42 43 44 45 46 47 48 49 50)
fi

bash scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh "${SEEDS[@]}"

RUNS=()
for seed in "${SEEDS[@]}"; do
  RUNS+=("reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${DATE_TAG}")
done

"${PY:-python}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --out-dir "reports/aggregate/metpair_backbone_context_stronglatent_${#SEEDS[@]}seed_macro_${DATE_TAG}"
