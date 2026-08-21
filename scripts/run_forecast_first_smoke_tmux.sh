#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="darts"
RUN_TAG=""
PREDICTOR_GPUS="0"
BASE_CFG="configs/base_forecast_first_smoke.yaml"
ENV_CFG="configs/env/windblown_case.yaml"
SENSOR_CFG="configs/sensors/windblown_sensors_complex.yaml"
ESTIMATOR_CFG="configs/estimator/kalman.yaml"
REWARD_CFG="configs/reward/lstm_aux_forecast_first_smoke.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --gpus)
      PREDICTOR_GPUS="$2"
      shift 2
      ;;
    --base-cfg)
      BASE_CFG="$2"
      shift 2
      ;;
    --env-cfg)
      ENV_CFG="$2"
      shift 2
      ;;
    --sensor-cfg)
      SENSOR_CFG="$2"
      shift 2
      ;;
    --estimator-cfg)
      ESTIMATOR_CFG="$2"
      shift 2
      ;;
    --reward-cfg)
      REWARD_CFG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  if [[ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/opt/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Conda init script not found, cannot activate env '${CONDA_ENV_NAME}'." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_NAME}"
fi

if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG="forecast_first_smoke_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p reports/logs
echo "RUN_TAG=${RUN_TAG}"
echo "BASE_CFG=${BASE_CFG}"
echo "ENV_CFG=${ENV_CFG}"
echo "SENSOR_CFG=${SENSOR_CFG}"
echo "REWARD_CFG=${REWARD_CFG}"

TRUTH_STEPS="$(
python - "${BASE_CFG}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(int(cfg.get("data", {}).get("truth_steps", 32768)))
PY
)"

python scripts/00_generate_business_data.py \
  --base_cfg "${BASE_CFG}" \
  --env_cfg "${ENV_CFG}" \
  --sensor_cfg "${SENSOR_CFG}" \
  --steps "${TRUTH_STEPS}" \
  --out data/generated/windblown_truth_${RUN_TAG}.csv \
  | tee "reports/logs/${RUN_TAG}_00_generate.log"

REWARD_RUN_ID="${RUN_TAG}_reward_model"
python scripts/00b_pretrain_reward_predictor.py \
  --truth_csv data/generated/windblown_truth_${RUN_TAG}.csv \
  --base_cfg "${BASE_CFG}" \
  --env_cfg "${ENV_CFG}" \
  --sensor_cfg "${SENSOR_CFG}" \
  --estimator_cfg "${ESTIMATOR_CFG}" \
  --reward_cfg "${REWARD_CFG}" \
  --run_id "${REWARD_RUN_ID}" \
  | tee "reports/logs/${RUN_TAG}_00b_reward_pretrain.log"

REWARD_ARTIFACT="reports/runs/${REWARD_RUN_ID}/reward_oracles.yaml"

declare -A SCHED_CFG=(
  [full_open]="configs/scheduler/full_open.yaml"
  [random]="configs/scheduler/random.yaml"
  [periodic]="configs/scheduler/periodic.yaml"
  [round_robin]="configs/scheduler/round_robin.yaml"
  [warmup_round_robin]="configs/scheduler/warmup_round_robin.yaml"
  [info_priority]="configs/scheduler/info_priority.yaml"
  [dqn]="configs/scheduler/dqn_smoke.yaml"
  [cmdp_dqn]="configs/scheduler/cmdp_dqn_smoke.yaml"
)

SCHED_ORDER=(full_open random periodic round_robin warmup_round_robin info_priority dqn cmdp_dqn)

for sched_name in "${SCHED_ORDER[@]}"; do
  RUN_ID="${RUN_TAG}_${sched_name}"
  CHECKPOINT_PATH="reports/runs/${RUN_ID}/scheduler_${sched_name}.pt"

  python scripts/01_train_rl_scheduler.py \
    --truth_csv data/generated/windblown_truth_${RUN_TAG}.csv \
    --base_cfg "${BASE_CFG}" \
    --env_cfg "${ENV_CFG}" \
    --sensor_cfg "${SENSOR_CFG}" \
    --estimator_cfg "${ESTIMATOR_CFG}" \
    --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
    --run_id "${RUN_ID}" \
    --reward_artifact "${REWARD_ARTIFACT}" \
    | tee "reports/logs/${RUN_ID}_train.log"

  EVAL_ARGS=()
  DATASET_ARGS=()
  if [[ "${sched_name}" == "dqn" || "${sched_name}" == "cmdp_dqn" ]]; then
    EVAL_ARGS+=(--checkpoint "${CHECKPOINT_PATH}")
    DATASET_ARGS+=(--checkpoint "${CHECKPOINT_PATH}")
  fi

  python scripts/02_evaluate_scheduler.py \
    --truth_csv data/generated/windblown_truth_${RUN_TAG}.csv \
    --base_cfg "${BASE_CFG}" \
    --env_cfg "${ENV_CFG}" \
    --sensor_cfg "${SENSOR_CFG}" \
    --estimator_cfg "${ESTIMATOR_CFG}" \
    --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
    --run_id "${RUN_ID}" \
    --reward_artifact "${REWARD_ARTIFACT}" \
    "${EVAL_ARGS[@]}" \
    | tee "reports/logs/${RUN_ID}_eval.log"

  python scripts/03_build_forecast_dataset.py \
    --truth_csv data/generated/windblown_truth_${RUN_TAG}.csv \
    --base_cfg "${BASE_CFG}" \
    --env_cfg "${ENV_CFG}" \
    --sensor_cfg "${SENSOR_CFG}" \
    --estimator_cfg "${ESTIMATOR_CFG}" \
    --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
    --run_id "${RUN_ID}" \
    --split final_test \
    --out_npz "data/processed/${RUN_ID}.npz" \
    "${DATASET_ARGS[@]}" \
    | tee "reports/logs/${RUN_ID}_dataset.log"
done

bash scripts/04_eval_frozen_predictors_multi_gpu.sh \
  --run-tag "${RUN_TAG}" \
  --reward-artifact "${REWARD_ARTIFACT}" \
  --gpus "${PREDICTOR_GPUS}" \
  --models tcn \
  --schedulers full_open,random,periodic,round_robin,warmup_round_robin,info_priority,dqn,cmdp_dqn \
  | tee "reports/logs/${RUN_TAG}_04_eval_frozen_predictors.log"

python scripts/05_evaluate_forecasts.py \
  --reports_dir reports/runs \
  --out_csv "reports/aggregate/metrics_forecast_all_${RUN_TAG}.csv" \
  --run_tag "${RUN_TAG}" \
  | tee "reports/logs/${RUN_TAG}_05_eval_forecast.log"

python scripts/06_posthoc_analysis.py \
  --metrics_csv "reports/aggregate/metrics_forecast_all_${RUN_TAG}.csv" \
  --out_dir "reports/aggregate/posthoc_${RUN_TAG}" \
  | tee "reports/logs/${RUN_TAG}_06_posthoc.log"

python scripts/09_generate_all_plots.py \
  --run-tag "${RUN_TAG}" \
  --target-set primary \
  --env-cfg "${ENV_CFG}" \
  --sensor-cfg "${SENSOR_CFG}" \
  --max-points 300 \
  --timeline-start 0 \
  --timeline-end 300 \
  | tee "reports/logs/${RUN_TAG}_09_generate_all_plots.log"

python scripts/11_plot_rl_training_diagnostics.py \
  --run-tag "${RUN_TAG}" \
  | tee "reports/logs/${RUN_TAG}_11_rl_training_plots.log"

echo "Forecast-first smoke experiment complete: ${RUN_TAG}"
