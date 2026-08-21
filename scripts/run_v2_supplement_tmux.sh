#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, inside tmux on the GPU server.
# This script intentionally separates training from aggregation so interrupted
# runs can be resumed without overwriting finished budget/seed directories.

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_GRID_DIR="${MAIN_GRID_DIR:-reports/v2_forecast_eval_grid_prior_kl1}"
SUPP_ROOT="${SUPP_ROOT:-reports/v2_supplement_experiments}"
ASSET_DIR="${ASSET_DIR:-reports/v2_supplement_assets}"

COMMON_ARGS=(
  --main-grid-dir "${MAIN_GRID_DIR}"
  --out-dir "${SUPP_ROOT}"
  --budgets 1.65 1.70 1.75
  --seeds 41 42 43 44 45 46 47 48 49 50
  --truth-steps 8192
  --oracle-rollout-steps 2400
  --oracle-epochs 18
  --total-timesteps 100000
  --n-steps 1024
  --batch-size 64
  --n-epochs 8
  --eval-steps 1024
  --eval-rollouts 6
  --device cuda
  --oracle-device cuda
  --oracle-inference-device cpu
)

echo "[supp] S1 main 10-seed grid"
"${PYTHON_BIN}" scripts/30_v2_run_supplement_experiments.py \
  --experiments s1 \
  "${COMMON_ARGS[@]}"

echo "[supp] A2 diagnostic sequence with D3"
"${PYTHON_BIN}" scripts/30_v2_run_supplement_experiments.py \
  --experiments a2 \
  --diagnostic-seed-count 5 \
  "${COMMON_ARGS[@]}"

echo "[supp] E1 calm/mixed/event condition evaluation"
"${PYTHON_BIN}" scripts/30_v2_run_supplement_experiments.py \
  --experiments e1 \
  "${COMMON_ARGS[@]}"

echo "[supp] Aggregate and draw currently available supplement assets"
"${PYTHON_BIN}" scripts/31_v2_build_supplement_assets.py \
  --grid-dirs "${MAIN_GRID_DIR}" \
  --table-dir reports/v2_paper_tables_prior_kl1 \
  --supp-root "${SUPP_ROOT}" \
  --out-dir "${ASSET_DIR}" \
  --budgets 1.65 1.70 1.75 \
  --seeds 41 42 43 44 45 46 47 48 49 50 \
  --bootstrap 10000

echo "[supp] done: ${ASSET_DIR}"
