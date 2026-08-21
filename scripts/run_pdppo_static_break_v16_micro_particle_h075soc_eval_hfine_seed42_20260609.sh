#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/scripts" ]; then
  cd "$SCRIPT_DIR"
else
  cd "$SCRIPT_DIR/.."
fi

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate darts 2>/dev/null || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/home/zhangzhuyu/.conda/envs/darts/bin/python
if [ ! -x "$PY" ]; then
  PY="$(command -v python)"
fi

SOURCE_RUN=reports/v31_static_break_v16_micro_particle_dwell12_ppo_seed42_h075_soc_20260609/raw/budget1p15_seed42
BASE_OUT=reports/v31_static_break_v16_micro_particle_h075soc_eval_hfine_seed42_20260609

for harvest in 0.81 0.82 0.83; do
  label="h${harvest/./}"
  out_dir="$BASE_OUT/$label/raw/budget1p15_seed42"
  mkdir -p "$BASE_OUT/$label/logs" "$out_dir"
  "$PY" scripts/64_v31_eval_saved_run_operational_baselines.py \
    --source-run-dir "$SOURCE_RUN" \
    --out-dir "$out_dir" \
    --device cpu \
    --oracle-device cpu \
    --env-min-dwell-steps 12 \
    --env-harvest-per-step "$harvest" \
    --eval-duty-constrained-baselines \
    --baseline-duty-hard-low 0.12 \
    --baseline-duty-hard-high 0.75 \
    --baseline-duty-hard-score 12 \
    --baseline-duty-score-feedback 2.5 \
    --skip-rollout-evaluation \
    > "$BASE_OUT/$label/logs/replay.log" 2>&1
done
