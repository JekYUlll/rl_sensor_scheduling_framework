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

OUT_DIR=reports/v31_static_break_v17_particle_heavy_dwell12_ppo_seed45_budget_bracket_h082_soc_oraclegreedy_eventeval_20260611

"$PY" scripts/59_v31_split_protocol_grid.py \
  --out-dir "$OUT_DIR" \
  --sensor-cfg configs/sensors/windblown_sensors_physical_event_v16_surface_boundary.yaml \
  --budgets 1.05 1.20 \
  --seeds 45 \
  --workers 2 \
  --gpu-ids 1,3 \
  --startup-peak-budget 1.55 \
  --truth-steps 60000 \
  --freq-s 3600 \
  --event-coverage 0.34 \
  --min-duration 12 \
  --max-duration 28 \
  --min-gap 4 \
  --lead-steps 6 \
  --flux-wind-exponent 4.0 \
  --event-microstructure-sigma 0.65 \
  --event-microstructure-alpha 0.20 \
  --event-microstructure-diameter-scale 0.16 \
  --event-microstructure-velocity-scale 1.50 \
  --event-particle-microstructure-correlation 0.00 \
  --oracle-rollout-steps 2400 \
  --oracle-rollouts-per-policy 4 \
  --oracle-epochs 8 \
  --oracle-batch-size 512 \
  --oracle-device auto \
  --oracle-inference-device cpu \
  --total-timesteps 40000 \
  --n-steps 512 \
  --batch-size 64 \
  --n-epochs 10 \
  --train-episode-len 512 \
  --event-gated-actor \
  --event-start-prob 0.90 \
  --event-reward-multiplier 3.0 \
  --soc-aux-horizon 32 \
  --soc-aux-coef 0.03 \
  --lambda-warmup-abort 1.00 \
  --lambda-switch 0.002 \
  --energy-account \
  --energy-capacity 180 \
  --initial-energy 180 \
  --harvest-per-step 0.82 \
  --reserve-energy 20 \
  --soc-soft-penalty-buffer 40 \
  --lambda-soc-soft-penalty 0.08 \
  --lambda-duty-balance 0.8 \
  --duty-balance-low 0.12 \
  --duty-balance-high 0.75 \
  --duty-score-feedback 2.5 \
  --duty-hard-guard \
  --duty-hard-low 0.12 \
  --duty-hard-high 0.75 \
  --duty-hard-score 12 \
  --min-dwell-steps 12 \
  --awbc-coef 0.40 \
  --awbc-label-stride 1 \
  --awbc-teacher-mode oracle_greedy \
  --prior-kl-coef 0.0 \
  --greedy-lookahead-steps 4 \
  --no-use-candidate-prior \
  --candidate-prior-scale 0.0 \
  --candidate-prior-steps 512 \
  --candidate-prior-rollouts 4 \
  --static-selection-steps 512 \
  --static-selection-rollouts 4 \
  --eval-steps 1024 \
  --eval-rollouts 4 \
  --eval-start-selection event_transport_rich \
  --eval-selection-stride 64 \
  --ent-coef 0.001 \
  --target-weights 0.03 0.03 0.10 0.01 0.01 0.0 16.0 22.0 22.0 \
  --target-scales 5.0 5.0 5.0 1.0 1.0 100.0 0.0001 0.2 5.0 \
  --max-active 4 \
  --eval-duty-constrained-baselines \
  --baseline-duty-hard-low 0.12 \
  --baseline-duty-hard-high 0.75 \
  --baseline-duty-hard-score 12 \
  --baseline-duty-score-feedback 2.5 \
  --bonferroni-family 1 \
  --skip-rollout-evaluation \
  --skip-collect

"$PY" scripts/65_v31_collect_operational_pdppo.py \
  --base-dir "$OUT_DIR" \
  --budget-label budget1p05 \
  --seeds 45 \
  --out-name v17_particle_heavy_b1p05_seed45_h082_soc_oraclegreedy_eventeval_summary.csv

"$PY" scripts/65_v31_collect_operational_pdppo.py \
  --base-dir "$OUT_DIR" \
  --budget-label budget1p20 \
  --seeds 45 \
  --out-name v17_particle_heavy_b1p20_seed45_h082_soc_oraclegreedy_eventeval_summary.csv
