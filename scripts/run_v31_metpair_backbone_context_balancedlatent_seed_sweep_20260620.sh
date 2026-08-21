#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_balancedlatent}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxbl}"
export DATE_TAG="${DATE_TAG:-20260620}"

# Moderate latent strengthening: enough to repair near-miss seeds 45/46 in the
# context branch, but avoids the strong-latent surface-temperature shortcut that
# made seed41 prefer a fixed surface_temp_ir static policy.
export EVENT_COVERAGE="${EVENT_COVERAGE:-0.68}"
export SUBTYPE_LATENT_ALPHA="${SUBTYPE_LATENT_ALPHA:-0.50}"
export PARTICLE_LATENT_DIAMETER_SCALE_MM="${PARTICLE_LATENT_DIAMETER_SCALE_MM:-0.30}"
export PARTICLE_LATENT_VELOCITY_SCALE_MS="${PARTICLE_LATENT_VELOCITY_SCALE_MS:-4.5}"
export FLUX_LATENT_SIGMA="${FLUX_LATENT_SIGMA:-2.6}"
export THERMAL_LATENT_SURFACE_SCALE_C="${THERMAL_LATENT_SURFACE_SCALE_C:-4.8}"
export FLUX_MULTIPLIER="${FLUX_MULTIPLIER:-8.0}"
export THERMAL_SURFACE_DROP_C="${THERMAL_SURFACE_DROP_C:-4.0}"
export THERMAL_AIR_TEMP_DROP_C="${THERMAL_AIR_TEMP_DROP_C:-1.2}"
export ORACLE_SUBTYPE_TEACHER_REPEAT="${ORACLE_SUBTYPE_TEACHER_REPEAT:-16}"
export AWBC_COEF="${AWBC_COEF:-1.0}"
export BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-2.5}"
export SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-2.5}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(41 45 46)
fi

bash scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh "${SEEDS[@]}"

RUNS=()
for seed in "${SEEDS[@]}"; do
  RUNS+=("reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${DATE_TAG}")
done

"${PY:-python}" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --out-dir "reports/aggregate/metpair_backbone_context_balancedlatent_${#SEEDS[@]}seed_macro_${DATE_TAG}"
