#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2}"
export BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolscbal2}"
export DATE_TAG="${DATE_TAG:-20260621}"

# SCENEBAL-2 is a simulator/data-balance pivot triggered by the pre-fixed
# conf0.5 confirmation wave: seed122 exposed a met_station_core+fc4_flux true
# fixed-static ordinary-step shortcut while replay and macro gates remained
# clean. Keep PPO and the met+one-specialist geometry; rebalance the synthetic
# event contract so particle/thermal states are less inferable from met+fc4.
export PARTICLE_PROB="${PARTICLE_PROB:-0.40}"
export FLUX_PROB="${FLUX_PROB:-0.20}"
export THERMAL_PROB="${THERMAL_PROB:-0.40}"

export SUBTYPE_LATENT_ALPHA="${SUBTYPE_LATENT_ALPHA:-0.65}"
export PARTICLE_LATENT_DIAMETER_SCALE_MM="${PARTICLE_LATENT_DIAMETER_SCALE_MM:-0.62}"
export PARTICLE_LATENT_VELOCITY_SCALE_MS="${PARTICLE_LATENT_VELOCITY_SCALE_MS:-8.2}"
export FLUX_MULTIPLIER="${FLUX_MULTIPLIER:-2.2}"
export FLUX_LATENT_SIGMA="${FLUX_LATENT_SIGMA:-0.0}"
export FLUX_LATENT_LINEAR_SCALE="${FLUX_LATENT_LINEAR_SCALE:-0.0014}"
export FLUX_LATENT_LINEAR_OFFSET="${FLUX_LATENT_LINEAR_OFFSET:-1.25}"
export FLUX_LATENT_LINEAR_CLIP="${FLUX_LATENT_LINEAR_CLIP:-2.4}"
export THERMAL_LATENT_SURFACE_SCALE_C="${THERMAL_LATENT_SURFACE_SCALE_C:-6.4}"
export THERMAL_SURFACE_DROP_C="${THERMAL_SURFACE_DROP_C:-5.8}"
export THERMAL_AIR_TEMP_DROP_C="${THERMAL_AIR_TEMP_DROP_C:-2.2}"

export EVENT_SUBTYPE_PARTICLE_REWARD_MULTIPLIER="${EVENT_SUBTYPE_PARTICLE_REWARD_MULTIPLIER:-1.55}"
export EVENT_SUBTYPE_FLUX_REWARD_MULTIPLIER="${EVENT_SUBTYPE_FLUX_REWARD_MULTIPLIER:-0.95}"
export EVENT_SUBTYPE_THERMAL_REWARD_MULTIPLIER="${EVENT_SUBTYPE_THERMAL_REWARD_MULTIPLIER:-1.65}"

export TARGET_WEIGHTS="${TARGET_WEIGHTS:-0.10 0.26 0.08 0.0 0.0 0.0 4.5 42.0 42.0}"
export SUBTYPE_PARTICLE_TARGET_WEIGHTS="${SUBTYPE_PARTICLE_TARGET_WEIGHTS:-0.0 0.0 0.08 0.0 0.0 0.0 0.5 105.0 105.0}"
export SUBTYPE_FLUX_TARGET_WEIGHTS="${SUBTYPE_FLUX_TARGET_WEIGHTS:-0.0 0.0 0.04 0.0 0.0 0.0 7.0 5.0 5.0}"
export SUBTYPE_THERMAL_TARGET_WEIGHTS="${SUBTYPE_THERMAL_TARGET_WEIGHTS:-0.22 85.0 0.08 0.0 0.0 0.0 0.2 1.0 1.0}"

bash scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal1_seed_sweep_20260621.sh "$@"
