#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
DATE_TAG="${DATE_TAG:-20260718corrected24}"
GPU_IDS_TEXT="${GPU_IDS:-1 2 3 4}"
SEEDS_TEXT="${SEEDS:-$(seq -s ' ' 117 140)}"
LOG_DIR="${LOG_DIR:-logs/matched_dqn_${DATE_TAG}}"
WAIT_SESSION="${WAIT_SESSION:-}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
TRAIN_ORACLE_DEVICE="${TRAIN_ORACLE_DEVICE:-cpu}"
EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-${TRAIN_ORACLE_DEVICE}}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
CPU_THREADS="${CPU_THREADS:-0}"
SOURCE_PREFIX="v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2"

read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_TEXT"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"
if [[ "${#GPU_IDS_ARRAY[@]}" -eq 0 || "${#SEEDS_ARRAY[@]}" -eq 0 ]]; then
  echo "At least one GPU and one seed are required" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"

if [[ -n "$WAIT_SESSION" ]]; then
  echo "[matched-dqn-sweep] waiting for tmux session ${WAIT_SESSION}"
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    sleep 60
  done
fi

echo "[matched-dqn-sweep] date=${DATE_TAG} seeds=${SEEDS_ARRAY[*]} gpus=${GPU_IDS_ARRAY[*]}"
pids=()
for worker_idx in "${!GPU_IDS_ARRAY[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS_ARRAY[$worker_idx]}"
    for seed_idx in "${!SEEDS_ARRAY[@]}"; do
      if (( seed_idx % ${#GPU_IDS_ARRAY[@]} != worker_idx )); then
        continue
      fi
      seed="${SEEDS_ARRAY[$seed_idx]}"
      source_dir="reports/${SOURCE_PREFIX}_seed${seed}_h075ctxolscbal2_20260621"
      out_dir="reports/v31_scenebal2_matched_dqn_seed${seed}_h075_${DATE_TAG}"
      if [[ -f "${out_dir}/v31_matched_dqn_metrics.csv" \
            && -f "${out_dir}/v31_matched_dqn_metadata.json" \
            && -f "${out_dir}/rollout_dqn.npz" ]]; then
        echo "[worker] complete artifact set exists; skip seed=${seed}"
        continue
      fi
      echo "[worker] start worker=${worker_idx} gpu=${CUDA_VISIBLE_DEVICES} seed=${seed} time=$(date -Is)"
      extra_args=()
      if [[ "$SKIP_TRAINING" == "1" ]]; then
        extra_args+=(--skip-training)
      fi
      "$PY" scripts/89_v31_train_matched_dqn.py \
        --source-run-dir "$source_dir" \
        --out-dir "$out_dir" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
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
        --oracle-device "$TRAIN_ORACLE_DEVICE" \
        --eval-oracle-device "$EVAL_ORACLE_DEVICE" \
        --cpu-threads "$CPU_THREADS" \
        "${extra_args[@]}"
      echo "[worker] done worker=${worker_idx} gpu=${CUDA_VISIBLE_DEVICES} seed=${seed} time=$(date -Is)"
    done
  ) > "${LOG_DIR}/worker${worker_idx}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[matched-dqn-sweep] failed; inspect ${LOG_DIR}" >&2
  exit "$status"
fi
echo "[matched-dqn-sweep] complete time=$(date -Is)"
