#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, inside tmux on the GPU server.
# This parallel runner resumes remaining supplementary tasks and assigns one
# training/evaluation subprocess per listed GPU. Finished directories are skipped.

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_GRID_DIR="${MAIN_GRID_DIR:-reports/v2_forecast_eval_grid_prior_kl1}"
SUPP_ROOT="${SUPP_ROOT:-reports/v2_supplement_experiments}"
ASSET_DIR="${ASSET_DIR:-reports/v2_supplement_assets}"
LOG_DIR="${LOG_DIR:-reports/logs/v2_supplement_parallel}"
GPUS=(${GPUS:-0 1 2 3 4 5})

"${PYTHON_BIN}" scripts/33_v2_run_supplement_parallel.py \
  --stages s1 a2 e1 aggregate \
  --main-grid-dir "${MAIN_GRID_DIR}" \
  --out-dir "${SUPP_ROOT}" \
  --asset-dir "${ASSET_DIR}" \
  --log-dir "${LOG_DIR}" \
  --gpus "${GPUS[@]}" \
  --budgets 1.65 1.70 1.75 \
  --seeds 41 42 43 44 45 46 47 48 49 50 \
  --truth-steps 8192 \
  --oracle-rollout-steps 2400 \
  --oracle-epochs 18 \
  --total-timesteps 100000 \
  --n-steps 1024 \
  --batch-size 64 \
  --n-epochs 8 \
  --eval-steps 1024 \
  --eval-rollouts 6 \
  --device cuda \
  --oracle-device cuda \
  --oracle-inference-device cpu \
  --bootstrap 10000
