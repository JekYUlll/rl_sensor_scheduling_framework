#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-20260718pilot}"
WAIT_SESSION="${WAIT_SESSION:-pdppo_matched_reward_pilot_20260718}"
GPU_IDS_TEXT="${GPU_IDS:-0 1}"
SEEDS_TEXT="${SEEDS:-117 118}"
LOG_DIR="${LOG_DIR:-logs/matched_dqn_${DATE_TAG}}"
SOURCE_PREFIX="v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2"

read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_TEXT"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"
if (( ${#GPU_IDS_ARRAY[@]} < ${#SEEDS_ARRAY[@]} )); then
  echo "Provide at least one GPU per pilot seed" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"

if [[ -n "$WAIT_SESSION" ]]; then
  echo "[matched-dqn] waiting for tmux session ${WAIT_SESSION}"
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    sleep 60
  done
fi

pids=()
for idx in "${!SEEDS_ARRAY[@]}"; do
  seed="${SEEDS_ARRAY[$idx]}"
  gpu="${GPU_IDS_ARRAY[$idx]}"
  source_dir="reports/${SOURCE_PREFIX}_seed${seed}_h075ctxolscbal2_20260621"
  out_dir="reports/v31_scenebal2_matched_dqn_seed${seed}_h075_${DATE_TAG}"
  if [[ -f "${out_dir}/v31_matched_dqn_metrics.csv" ]]; then
    echo "[matched-dqn] complete artifact exists; skip seed=${seed}"
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[matched-dqn] start seed=${seed} gpu=${gpu} time=$(date -Is)"
    "$PY" scripts/89_v31_train_matched_dqn.py \
      --source-run-dir "$source_dir" \
      --out-dir "$out_dir" \
      --total-timesteps 200000 \
      --replay-size 100000 \
      --learning-starts 5000 \
      --batch-size 128 \
      --train-freq 4 \
      --gradient-steps 1 \
      --target-update-interval 1000 \
      --learning-rate 1e-4 \
      --gamma 0.99 \
      --n-step-return 3 \
      --hidden-dim 128 \
      --exploration-fraction 0.20 \
      --exploration-final-eps 0.05 \
      --device cuda \
      --oracle-device cpu
    echo "[matched-dqn] done seed=${seed} gpu=${gpu} time=$(date -Is)"
  ) > "${LOG_DIR}/seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[matched-dqn] pilot failed; inspect ${LOG_DIR}" >&2
  exit "$status"
fi
echo "[matched-dqn] pilot complete time=$(date -Is)"

