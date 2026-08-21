#!/usr/bin/env bash
set -euo pipefail

# Replay the frozen clean PD-PPO checkpoints and validation-selected static
# masks over every scoreable epoch in the held-out final partition.  This is a
# sensitivity evaluation only: no policy, oracle, normalizer, or mask is fitted.

ROOT="${PD_PPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED_START="${SEED_START:-117}"
SEED_END="${SEED_END:-140}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
MAX_JOBS="${MAX_JOBS:-5}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
THREADS_PER_JOB="${THREADS_PER_JOB:-8}"
OUT_SUBDIR="${OUT_SUBDIR:-full_final_partition_replay}"
AGGREGATE_DIR="${AGGREGATE_DIR:-reports/aggregate/pdppo_full_final_partition_24seed_20260718}"
RUN_PREFIX="v31_scenebal2_matched_reward_forecast_noexactevent_seed"
RUN_SUFFIX="_h075forecastctrl_20260718cleanpilot"

cd "$ROOT"
read -r -a GPUS <<< "$GPU_LIST"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_LIST must contain at least one GPU index" >&2
  exit 2
fi

active_jobs() {
  jobs -pr | wc -l
}

run_seed() {
  local seed="$1"
  local gpu="$2"
  local run_dir="reports/${RUN_PREFIX}${seed}${RUN_SUFFIX}"
  local out_dir="${run_dir}/${OUT_SUBDIR}"
  local log_path="reports/full_final_partition_seed${seed}_20260718.log"

  if [[ ! -f "${run_dir}/custom_ppo.pt" || ! -f "${run_dir}/v2_ppo_metadata.json" ]]; then
    echo "seed=${seed} missing frozen source run: ${run_dir}" >&2
    return 1
  fi
  if [[ -s "${out_dir}/v2_custom_ppo_metrics.csv" && \
        -s "${out_dir}/rollout_custom_ppo.npz" && \
        -s "${out_dir}/rollout_validation_selected_static.npz" ]]; then
    echo "seed=${seed} already complete"
    return 0
  fi

  mkdir -p "$out_dir"
  echo "seed=${seed} gpu=${gpu} start"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS_PER_JOB" TORCH_NUM_THREADS="$THREADS_PER_JOB" \
    python scripts/64_v31_eval_saved_run_operational_baselines.py \
      --source-run-dir "$run_dir" \
      --out-dir "$out_dir" \
      --device "$EVAL_DEVICE" \
      --oracle-device "$EVAL_DEVICE" \
      --eval-full-final-partition \
      --primary-only \
      --skip-rollout-evaluation \
      > "$log_path" 2>&1
  echo "seed=${seed} gpu=${gpu} complete"
}

job_index=0
for seed in $(seq "$SEED_START" "$SEED_END"); do
  while (( $(active_jobs) >= MAX_JOBS )); do
    wait -n
  done
  gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  run_seed "$seed" "$gpu" &
  job_index=$((job_index + 1))
done
wait

seed_args=()
for seed in $(seq "$SEED_START" "$SEED_END"); do
  seed_args+=("$seed")
done

python scripts/86_v31_collect_validation_frozen_macro.py \
  --run-glob "reports/${RUN_PREFIX}*${RUN_SUFFIX}" \
  --seeds "${seed_args[@]}" \
  --router-eval-dir "$OUT_SUBDIR" \
  --out-dir "$AGGREGATE_DIR" \
  --bootstrap-samples 100000

echo "aggregate=${AGGREGATE_DIR}"
