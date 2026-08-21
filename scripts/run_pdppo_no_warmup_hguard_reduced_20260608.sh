#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/scripts" ]; then
  cd "$SCRIPT_DIR"
else
  cd "$SCRIPT_DIR/.."
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/home/zhangzhuyu/.conda/envs/darts/bin/python

"$PY" scripts/59_v31_split_protocol_grid.py \
  --out-dir reports/v31_split_protocol_no_warmup_hguard_reduced \
  --sensor-cfg configs/sensors/windblown_sensors_balanced_no_warmup.yaml \
  --budgets 1.70 \
  --seeds 41 42 43 \
  --workers 1 \
  --gpu-ids 5 \
  --total-timesteps 40000 \
  --lambda-warmup-abort 0.0 \
  --lambda-duty-balance 0.8 \
  --duty-balance-low 0.12 \
  --duty-balance-high 0.85 \
  --duty-score-feedback 2.5 \
  --duty-hard-guard \
  --duty-hard-low 0.12 \
  --duty-hard-high 0.85 \
  --duty-hard-score 12 \
  --awbc-coef 0.02 \
  --prior-kl-coef 0.05 \
  --candidate-prior-scale 0.5 \
  --ent-coef 0.003 \
  --eval-duty-constrained-baselines \
  --baseline-duty-hard-low 0.12 \
  --baseline-duty-hard-high 0.85 \
  --baseline-duty-hard-score 12 \
  --baseline-duty-score-feedback 2.5 \
  --bonferroni-family 3
