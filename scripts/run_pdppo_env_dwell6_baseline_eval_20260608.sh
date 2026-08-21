#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/scripts" ]; then
  cd "$SCRIPT_DIR"
else
  cd "$SCRIPT_DIR/.."
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-8}"

PY=/home/zhangzhuyu/.conda/envs/darts/bin/python
SRC_ROOT=reports/v31_split_protocol_no_warmup_hguard_reduced/raw
OUT_ROOT=reports/v31_env_dwell6_operational_eval

for seed in 41 42 43; do
  "$PY" scripts/64_v31_eval_saved_run_operational_baselines.py \
    --source-run-dir "$SRC_ROOT/budget1p70_seed${seed}" \
    --out-dir "$OUT_ROOT/no_warmup_hguard_seed${seed}" \
    --device cpu \
    --oracle-device cpu \
    --env-min-dwell-steps 6 \
    --eval-duty-constrained-baselines \
    --baseline-duty-hard-low 0.12 \
    --baseline-duty-hard-high 0.85 \
    --baseline-duty-hard-score 12 \
    --baseline-duty-score-feedback 2.5 \
    --skip-rollout-evaluation
done
