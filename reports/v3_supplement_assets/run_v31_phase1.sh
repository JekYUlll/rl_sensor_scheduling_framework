#!/usr/bin/env bash
set -euo pipefail
cd /home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY=/home/zhangzhuyu/.conda/envs/darts/bin/python
{
  echo "[start] $(date)"
  "$PY" scripts/37_v2_run_v31_phase1.py --experiment all --workers 3 --gpu-ids 1,4,5
  echo "[collect] $(date)"
  "$PY" scripts/38_v2_collect_v31_phase1.py
  echo "[done] $(date)"
} 2>&1 | tee reports/v3_supplement_assets/phase1_driver.log
