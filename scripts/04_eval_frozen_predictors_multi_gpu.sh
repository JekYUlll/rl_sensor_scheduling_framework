#!/usr/bin/env bash
set -euo pipefail

RUN_TAG=""
REWARD_ARTIFACT=""
GPUS=""
SKIP_EXISTING=1
MODELS="tcn,lstm,transformer"
SCHEDULERS="full_open,random,periodic,round_robin,info_priority,dqn,cmdp_dqn,ppo"

cleanup_children() {
  local pids
  pids="$(jobs -pr || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
  fi
}

trap cleanup_children INT TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --reward-artifact)
      REWARD_ARTIFACT="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --schedulers)
      SCHEDULERS="$2"
      shift 2
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_TAG}" ]]; then
  echo "--run-tag is required" >&2
  exit 1
fi
if [[ -z "${REWARD_ARTIFACT}" ]]; then
  echo "--reward-artifact is required" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p reports/logs

if [[ "${GPUS}" == "cpu" || "${GPUS}" == "none" ]]; then
  GPUS=""
  GPU_LIST=()
elif [[ -z "${GPUS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
    if [[ "${GPU_COUNT}" -gt 0 ]]; then
      GPUS="$(seq -s, 0 $((GPU_COUNT - 1)))"
    fi
  fi
else
  IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
fi

if [[ -z "${GPUS}" ]]; then
  GPU_LIST=()
else
  IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
fi
IFS=',' read -r -a SCHEDULER_LIST <<< "${SCHEDULERS}"
IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"
MAX_JOBS="${#GPU_LIST[@]}"

echo "RUN_TAG=${RUN_TAG}"
echo "REWARD_ARTIFACT=${REWARD_ARTIFACT}"
echo "GPUS=${GPUS}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "SCHEDULERS=${SCHEDULERS}"
echo "MODELS=${MODELS}"

run_eval_job() {
  local device="$1"
  local sched_name="$2"
  local model_name="$3"
  local log_suffix="$4"
  local dataset_run_id="${RUN_TAG}_${sched_name}"
  local run_id="${dataset_run_id}_pred_${model_name}"
  local out_dir="reports/runs/${run_id}"
  local log_path="reports/logs/${run_id}_${log_suffix}.log"

  if [[ ! -f "data/processed/${dataset_run_id}.npz" ]]; then
    echo "[missing-dataset] data/processed/${dataset_run_id}.npz" >&2
    return 1
  fi
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${out_dir}/metrics_forecast.csv" ]]; then
    echo "[skip] ${run_id}"
    return 0
  fi
  echo "[run][device=${device}] ${run_id}"
  python scripts/04_eval_frozen_predictors.py \
    --series_npz "data/processed/${dataset_run_id}.npz" \
    --reward_artifact "${REWARD_ARTIFACT}" \
    --model_name "${model_name}" \
    --run_id "${run_id}" \
    > "${log_path}" 2>&1
}

launch_gpu_job() {
  local gpu="$1"
  local sched_name="$2"
  local model_name="$3"
  echo "[launch][gpu=${gpu}] ${RUN_TAG}_${sched_name}_pred_${model_name}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    run_eval_job "cuda:0" "${sched_name}" "${model_name}" "eval_frozen"
  ) &
}

active_jobs=0
job_index=0
for sched_name in "${SCHEDULER_LIST[@]}"; do
  for model_name in "${MODEL_LIST[@]}"; do
    if [[ "${MAX_JOBS}" -lt 1 ]]; then
      run_eval_job "cpu" "${sched_name}" "${model_name}" "eval_frozen"
      continue
    fi
    while [[ "${active_jobs}" -ge "${MAX_JOBS}" ]]; do
      wait -n
      active_jobs=$((active_jobs - 1))
    done
    gpu="${GPU_LIST[$((job_index % MAX_JOBS))]}"
    launch_gpu_job "${gpu}" "${sched_name}" "${model_name}"
    active_jobs=$((active_jobs + 1))
    job_index=$((job_index + 1))
  done
done

while [[ "${active_jobs}" -gt 0 ]]; do
  wait -n
  active_jobs=$((active_jobs - 1))
done

wait

echo "Frozen predictor evaluation done for ${RUN_TAG}."
