#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxol}"
export DATE_TAG="${DATE_TAG:-20260620}"

# Orthogonal-linear generator branch:
# - keep the contextual subtype task,
# - replace unstable exponential flux latent amplification with a bounded
#   linear flux-latent term,
# - reduce thermal latent strength to prevent fixed surface_temp_ir from
#   becoming a cross-regime shortcut.
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
export ORACLE_SUBTYPE_TEACHER_REPEAT="${ORACLE_SUBTYPE_TEACHER_REPEAT:-16}"
export AWBC_COEF="${AWBC_COEF:-1.0}"
export BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-2.5}"
export SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-2.5}"

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
  --out-dir "reports/aggregate/metpair_backbone_context_ortholinear_${#SEEDS[@]}seed_macro_${DATE_TAG}"
