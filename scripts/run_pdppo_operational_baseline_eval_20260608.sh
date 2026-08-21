#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/64_v31_eval_saved_run_operational_baselines.py" ]; then
  cd "$SCRIPT_DIR/.."
else
  cd "$SCRIPT_DIR"
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCH_NUM_THREADS=8

PY=/home/zhangzhuyu/.conda/envs/darts/bin/python
COMMON=(
  --device cpu
  --oracle-device cpu
  --eval-duty-constrained-baselines
  --baseline-duty-hard-low 0.12
  --baseline-duty-hard-high 0.85
  --baseline-duty-hard-score 12
  --baseline-duty-score-feedback 2.5
  --skip-rollout-evaluation
)

"$PY" scripts/64_v31_eval_saved_run_operational_baselines.py \
  --source-run-dir reports/v31_static_break_duty_pilot/v10_b0p65_particle_energy_cov_hguard_l08h90s12_dfb2p5_seed41 \
  --out-dir reports/v31_operational_baseline_eval/v10_b0p65_hguard_seed41 \
  "${COMMON[@]}"

"$PY" scripts/64_v31_eval_saved_run_operational_baselines.py \
  --source-run-dir reports/v31_static_break_duty_pilot/v10_b0p65_particle_energy_cov_hguard_l08h90s12_dfb2p5_seed42 \
  --out-dir reports/v31_operational_baseline_eval/v10_b0p65_hguard_seed42 \
  "${COMMON[@]}"

"$PY" scripts/64_v31_eval_saved_run_operational_baselines.py \
  --source-run-dir reports/v31_static_break_duty_pilot/v10_b0p65_particle_energy_cov_hguard_l12h85s12_dfb2p5_lam0p8_awbc0p02_kl0p05_prior0p5_seed43 \
  --out-dir reports/v31_operational_baseline_eval/v10_b0p65_h85_weakprior_seed43 \
  "${COMMON[@]}"
