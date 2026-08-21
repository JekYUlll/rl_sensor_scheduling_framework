#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <forecast|aoi|uncertainty> [seed ...]" >&2
  exit 2
fi

MODE="$1"
shift
case "$MODE" in
  forecast|aoi|uncertainty) ;;
  *)
    echo "Unsupported reward control: ${MODE}" >&2
    exit 2
    ;;
esac

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(117 118)
fi

# Every control reuses the authoritative SCENEBAL-2 assets for its seed. The
# source validator checks byte-identical truth, candidate masks, partitions,
# constraints, frozen forecaster, static normalisers, and validation selection.
export CONTROL_SOURCE_RUN_PREFIX="v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2"
export CONTROL_SOURCE_BUDGET_LABEL="h075ctxolscbal2"
export CONTROL_SOURCE_DATE_TAG="20260621"

export RUN_PREFIX="v31_scenebal2_matched_reward_${MODE}_noexactevent"
export BUDGET_LABEL="h075${MODE}ctrl"
export DATE_TAG="${DATE_TAG:-20260718}"
export REWARD_PROXY_MODE="$MODE"

# Correct the online-observability contract. Training-only subtype labels stay
# available to the existing auxiliary losses, while actor and critic execution
# use the station-side alert columns already present in the source truth.
export INCLUDE_EVENT_FLAG_IN_STATE=0
export INCLUDE_ALERT_CONTEXT_FEATURES=0
export INCLUDE_OBSERVABLE_REGIME_BELIEF=1
export EVENT_AWARE_CRITIC=1
export EVENT_GATED_ACTOR=1
# The auxiliary subtype head may shape the shared representation during
# training, but the primary policy must execute through the masked PPO actor.
# Hard subtype-to-action routing is a separate hybrid policy and is disabled.
export SUBTYPE_ROUTER_ENABLED="${SUBTYPE_ROUTER_ENABLED:-0}"
export CONTEXT_ENCODER="${CONTEXT_ENCODER:-0}"
export CONTEXT_FEATURE_DIM="${CONTEXT_FEATURE_DIM:-0}"
export CONTEXT_HIDDEN_DIM="${CONTEXT_HIDDEN_DIM:-64}"
export CONTEXT_FUSION_MODE="${CONTEXT_FUSION_MODE:-concat}"
export CONTEXT_LAYER_NORM="${CONTEXT_LAYER_NORM:-0}"

export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
export N_STEPS="${N_STEPS:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export N_EPOCHS="${N_EPOCHS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export ENT_COEF="${ENT_COEF:-0.0075}"

# Make long seed sweeps safely resumable.  A run is complete only when the
# learned-policy metrics, provenance metadata, and held-out rollout all exist.
PENDING_SEEDS=()
for seed in "${SEEDS[@]}"; do
  run_dir="reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${DATE_TAG}"
  complete_run=0
  if [[ -f "${run_dir}/v2_custom_ppo_metrics.csv" \
        && -f "${run_dir}/v2_ppo_metadata.json" \
        && -f "${run_dir}/rollout_custom_ppo.npz" ]]; then
    complete_run=1
  fi
  if [[ "${complete_run}" -eq 1 ]]; then
    echo "[matched-reward] complete artifact set exists; skip mode=${MODE} seed=${seed}"
  elif [[ -f "${run_dir}/.matched_reward_in_progress" ]]; then
    echo "[matched-reward] waiting for active artifact writer mode=${MODE} seed=${seed}"
    while [[ -f "${run_dir}/.matched_reward_in_progress" ]]; do
      if [[ -f "${run_dir}/v2_custom_ppo_metrics.csv" \
            && -f "${run_dir}/v2_ppo_metadata.json" \
            && -f "${run_dir}/rollout_custom_ppo.npz" ]]; then
        complete_run=1
        break
      fi
      sleep 30
    done
    if [[ "${complete_run}" -eq 1 ]]; then
      echo "[matched-reward] active writer completed; skip mode=${MODE} seed=${seed}"
    else
      PENDING_SEEDS+=("${seed}")
    fi
  else
    PENDING_SEEDS+=("${seed}")
  fi
done
if [[ "${#PENDING_SEEDS[@]}" -eq 0 ]]; then
  exit 0
fi
SEEDS=("${PENDING_SEEDS[@]}")

export SKIP_COLLECT=1
export SKIP_ROUTER_EVAL=1
export SKIP_REPLAY_GATE=1
export SKIP_BEHAVIOR_AUDIT=1

echo "[matched-reward] mode=${MODE} seeds=${SEEDS[*]} date=${DATE_TAG}"
bash scripts/run_v31_scenebal2_seed_sweep_20260621.sh "${SEEDS[@]}"
