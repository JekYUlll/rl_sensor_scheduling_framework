#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
PLAIN_TAG="${PLAIN_TAG:-20260718cleanpilot}"
CONTEXT_TAG="${CONTEXT_TAG:-20260718capilot}"
OUT_DIR="${OUT_DIR:-reports/aggregate/pdppo_clean_method_gate_20260718}"
SEEDS=(117 118)

complete_run() {
  local tag="$1"
  local seed="$2"
  local run_dir="reports/v31_scenebal2_matched_reward_forecast_noexactevent_seed${seed}_h075forecastctrl_${tag}"
  [[ -f "${run_dir}/v2_custom_ppo_metrics.csv" \
     && -f "${run_dir}/v2_ppo_metadata.json" \
     && -f "${run_dir}/rollout_custom_ppo.npz" ]]
}

while true; do
  pending=0
  for tag in "$PLAIN_TAG" "$CONTEXT_TAG"; do
    for seed in "${SEEDS[@]}"; do
      if ! complete_run "$tag" "$seed"; then
        pending=$((pending + 1))
      fi
    done
  done
  if [[ "$pending" -eq 0 ]]; then
    break
  fi
  if ! tmux has-session -t pdppo_clean_norouter_pilot_20260718 2>/dev/null \
     && ! tmux has-session -t pdppo_ca_norouter_pilot_20260718 2>/dev/null; then
    echo "[clean-gate] pilot sessions ended with ${pending} incomplete runs" >&2
    exit 1
  fi
  echo "[clean-gate] waiting pending=${pending} time=$(date -Is)"
  sleep 60
done

"$PY" scripts/92_v31_select_clean_pdppo.py \
  --reports-root reports \
  --plain-tag "$PLAIN_TAG" \
  --context-tag "$CONTEXT_TAG" \
  --seeds "${SEEDS[@]}" \
  --material-macro-improvement 0.005 \
  --out-dir "$OUT_DIR"

echo "[clean-gate] complete out=${OUT_DIR} time=$(date -Is)"
