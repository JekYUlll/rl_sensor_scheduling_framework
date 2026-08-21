#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
DATE_TAG="${DATE_TAG:-20260718pilot}"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2 3 4}"
SEEDS_TEXT="${SEEDS:-117 118}"
MODES_TEXT="${MODES:-forecast aoi uncertainty}"
LOG_DIR="${LOG_DIR:-logs/matched_reward_${DATE_TAG}}"

read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_TEXT"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"
read -r -a MODES_ARRAY <<< "$MODES_TEXT"
mkdir -p "$LOG_DIR"

JOBS=()
for mode in "${MODES_ARRAY[@]}"; do
  for seed in "${SEEDS_ARRAY[@]}"; do
    JOBS+=("${mode}:${seed}")
  done
done

echo "[matched-reward-pilot] date=${DATE_TAG} jobs=${JOBS[*]} gpus=${GPU_IDS_ARRAY[*]}"
pids=()
for worker_idx in "${!GPU_IDS_ARRAY[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS_ARRAY[$worker_idx]}"
    export PY DATE_TAG
    for job_idx in "${!JOBS[@]}"; do
      if (( job_idx % ${#GPU_IDS_ARRAY[@]} != worker_idx )); then
        continue
      fi
      IFS=: read -r mode seed <<< "${JOBS[$job_idx]}"
      echo "[worker] start worker=${worker_idx} gpu=${CUDA_VISIBLE_DEVICES} mode=${mode} seed=${seed} time=$(date -Is)"
      bash scripts/run_v31_matched_reward_controls_20260718.sh "$mode" "$seed"
      echo "[worker] done worker=${worker_idx} gpu=${CUDA_VISIBLE_DEVICES} mode=${mode} seed=${seed} time=$(date -Is)"
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
  echo "[matched-reward-pilot] failed; inspect ${LOG_DIR}" >&2
  exit "$status"
fi
echo "[matched-reward-pilot] complete time=$(date -Is)"

