#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="darts"
RUN_TAG=""
PREDICTOR_GPUS=""

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

REWARD_RUN_ID="${RUN_TAG}_reward_model"
python scripts/00b_pretrain_reward_predictor.py \
  --truth_csv data/generated/windblown_truth.csv \
  --env_cfg configs/env/windblown_case.yaml \
  --sensor_cfg configs/sensors/windblown_sensors.yaml \
  --estimator_cfg configs/estimator/kalman.yaml \
  --reward_cfg configs/reward/lstm_aux.yaml \
  --run_id "${REWARD_RUN_ID}" \
  | tee "reports/logs/${RUN_TAG}_00b_reward_pretrain.log"

REWARD_ARTIFACT="reports/runs/${REWARD_RUN_ID}/reward_predictor.pt"

declare -A SCHED_CFG=(
  [full_open]="configs/scheduler/full_open.yaml"
  [random]="configs/scheduler/random.yaml"
  [periodic]="configs/scheduler/periodic.yaml"
  [round_robin]="configs/scheduler/round_robin.yaml"
  [info_priority]="configs/scheduler/info_priority.yaml"
  [dqn]="configs/scheduler/dqn.yaml"
  [cmdp_dqn]="configs/scheduler/cmdp_dqn.yaml"
  [ppo]="configs/scheduler/ppo.yaml"
)

for sched_name in full_open random periodic round_robin info_priority dqn cmdp_dqn ppo; do
  RUN_ID="${RUN_TAG}_${sched_name}"
  CHECKPOINT_PATH="reports/runs/${RUN_ID}/scheduler_${sched_name}.pt"
  if [[ "${sched_name}" == "ppo" ]]; then
    CHECKPOINT_PATH="reports/runs/${RUN_ID}/scheduler_${sched_name}.zip"
  fi

  python scripts/01_train_rl_scheduler.py \
    --truth_csv data/generated/windblown_truth.csv \
    --env_cfg configs/env/windblown_case.yaml \
    --sensor_cfg configs/sensors/windblown_sensors.yaml \
    --estimator_cfg configs/estimator/kalman.yaml \
    --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
    --run_id "${RUN_ID}" \
    --reward_artifact "${REWARD_ARTIFACT}" \
    | tee "reports/logs/${RUN_ID}_train.log"

  if [[ "${sched_name}" == "dqn" || "${sched_name}" == "cmdp_dqn" || "${sched_name}" == "ppo" ]]; then
    python scripts/02_evaluate_scheduler.py \
      --truth_csv data/generated/windblown_truth.csv \
      --env_cfg configs/env/windblown_case.yaml \
      --sensor_cfg configs/sensors/windblown_sensors.yaml \
      --estimator_cfg configs/estimator/kalman.yaml \
      --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
      --run_id "${RUN_ID}" \
      --checkpoint "${CHECKPOINT_PATH}" \
      --reward_artifact "${REWARD_ARTIFACT}" \
      | tee "reports/logs/${RUN_ID}_eval.log"

    python scripts/03_build_forecast_dataset.py \
      --truth_csv data/generated/windblown_truth.csv \
      --env_cfg configs/env/windblown_case.yaml \
      --sensor_cfg configs/sensors/windblown_sensors.yaml \
      --estimator_cfg configs/estimator/kalman.yaml \
      --scheduler_cfg "${SCHED_CFG[$sched_name]}" \
      --run_id "${RUN_ID}" \
      --checkpoint "${CHECKPOINT_PATH}" \
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
      --reward_artifact "${REWARD_ARTIFACT}" \
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

PREDICTOR_CMD=(bash scripts/04_train_predictors_multi_gpu.sh --run-tag "${RUN_TAG}")
if [[ -n "${PREDICTOR_GPUS}" ]]; then
  PREDICTOR_CMD+=(--gpus "${PREDICTOR_GPUS}")
fi
"${PREDICTOR_CMD[@]}" | tee "reports/logs/${RUN_TAG}_04_train_predictors_multi_gpu.log"

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
  --max-points 300 \
  --timeline-start 0 \
  --timeline-end 300 \
  | tee "reports/logs/${RUN_TAG}_09_generate_all_plots.log"

python scripts/09_generate_all_plots.py \
  --run-tag "${RUN_TAG}" \
  --target snow_mass_flux_kg_m2_s \
  --target-set single \
  --max-points 300 \
  --timeline-start 0 \
  --timeline-end 300 \
  | tee "reports/logs/${RUN_TAG}_09b_generate_snow_flux_plots.log"

python scripts/10_posthoc_task_focus.py \
  --run-tag "${RUN_TAG}" \
  --env-cfg configs/env/windblown_case.yaml \
  | tee "reports/logs/${RUN_TAG}_10_posthoc_task_focus.log"

python scripts/11_plot_rl_training_diagnostics.py \
  --run-tag "${RUN_TAG}" \
  | tee "reports/logs/${RUN_TAG}_11_rl_training_plots.log"

python scripts/13_plot_legacy_style_summaries.py \
  --run-tag "${RUN_TAG}" \
  --env-cfg configs/env/windblown_case.yaml \
  --model informer \
  | tee "reports/logs/${RUN_TAG}_13_legacy_style_plots.log"

echo "DONE: ${RUN_TAG}"
echo "Logs: reports/logs/"
echo "Aggregate metrics: reports/aggregate/metrics_forecast_all_${RUN_TAG}.csv"
