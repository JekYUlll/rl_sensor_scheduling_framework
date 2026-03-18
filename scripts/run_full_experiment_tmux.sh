#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="darts"
RUN_TAG=""

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
  RUN_TAG="full_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p reports/logs
echo "RUN_TAG=${RUN_TAG}"

python scripts/00_generate_business_data.py \
  --env_cfg configs/env/windblown_case.yaml \
  --sensor_cfg configs/sensors/windblown_sensors.yaml \
  --steps 10000 \
  --out data/generated/windblown_truth.csv \
  | tee "reports/logs/${RUN_TAG}_00_generate.log"

declare -A SCHED_CFG=(
  [full_open]="configs/scheduler/full_open.yaml"
  [random]="configs/scheduler/random.yaml"
  [periodic]="configs/scheduler/periodic.yaml"
  [round_robin]="configs/scheduler/round_robin.yaml"
  [info_priority]="configs/scheduler/info_priority.yaml"
  [dqn]="configs/scheduler/dqn.yaml"
)

for sched_name in full_open random periodic round_robin info_priority dqn; do
  RUN_ID="${RUN_TAG}_${sched_name}"

  python scripts/01_train_rl_scheduler.py \
    --truth_csv data/generated/windblown_truth.csv \
    --env_cfg configs/env/windblown_case.yaml \
    --sensor_cfg configs/sensors/windblown_sensors.yaml \
    --estimator_cfg configs/estimator/kalman.yaml \
    --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
    --run_id "${RUN_ID}" \
    | tee "reports/logs/${RUN_ID}_train.log"

  if [[ "${sched_name}" == "dqn" ]]; then
    python scripts/02_evaluate_scheduler.py \
      --truth_csv data/generated/windblown_truth.csv \
      --env_cfg configs/env/windblown_case.yaml \
      --sensor_cfg configs/sensors/windblown_sensors.yaml \
      --estimator_cfg configs/estimator/kalman.yaml \
      --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
      --run_id "${RUN_ID}" \
      --checkpoint "reports/runs/${RUN_ID}/scheduler_dqn.pt" \
      | tee "reports/logs/${RUN_ID}_eval.log"

    python scripts/03_build_forecast_dataset.py \
      --truth_csv data/generated/windblown_truth.csv \
      --env_cfg configs/env/windblown_case.yaml \
      --sensor_cfg configs/sensors/windblown_sensors.yaml \
      --estimator_cfg configs/estimator/kalman.yaml \
      --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
      --run_id "${RUN_ID}" \
      --checkpoint "reports/runs/${RUN_ID}/scheduler_dqn.pt" \
      --out_npz "data/processed/${RUN_ID}.npz" \
      | tee "reports/logs/${RUN_ID}_dataset.log"
  else
    python scripts/02_evaluate_scheduler.py \
      --truth_csv data/generated/windblown_truth.csv \
      --env_cfg configs/env/windblown_case.yaml \
      --sensor_cfg configs/sensors/windblown_sensors.yaml \
      --estimator_cfg configs/estimator/kalman.yaml \
      --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
      --run_id "${RUN_ID}" \
      | tee "reports/logs/${RUN_ID}_eval.log"

    python scripts/03_build_forecast_dataset.py \
      --truth_csv data/generated/windblown_truth.csv \
      --env_cfg configs/env/windblown_case.yaml \
      --sensor_cfg configs/sensors/windblown_sensors.yaml \
      --estimator_cfg configs/estimator/kalman.yaml \
      --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
      --run_id "${RUN_ID}" \
      --out_npz "data/processed/${RUN_ID}.npz" \
      | tee "reports/logs/${RUN_ID}_dataset.log"
  fi

done

for sched_name in full_open random periodic round_robin info_priority dqn; do
  DATASET_RUN_ID="${RUN_TAG}_${sched_name}"
  for model_name in naive mlp lstm transformer informer tcn; do
    RUN_ID="${DATASET_RUN_ID}_pred_${model_name}"
    python scripts/04_train_predictors.py \
      --series_npz "data/processed/${DATASET_RUN_ID}.npz" \
      --predictor_cfg "configs/predictor/${model_name}.yaml" \
      --run_id "${RUN_ID}" \
      | tee "reports/logs/${RUN_ID}_train_predictor.log"
  done
done

python scripts/05_evaluate_forecasts.py \
  --reports_dir reports/runs \
  --out_csv "reports/aggregate/metrics_forecast_all_${RUN_TAG}.csv" \
  --run_tag "${RUN_TAG}" \
  | tee "reports/logs/${RUN_TAG}_05_eval_forecast.log"

python scripts/06_posthoc_analysis.py \
  --metrics_csv "reports/aggregate/metrics_forecast_all_${RUN_TAG}.csv" \
  --out_dir "reports/aggregate/posthoc_${RUN_TAG}" \
  | tee "reports/logs/${RUN_TAG}_06_posthoc.log"

echo "DONE: ${RUN_TAG}"
echo "Logs: reports/logs/"
echo "Aggregate metrics: reports/aggregate/metrics_forecast_all_${RUN_TAG}.csv"
