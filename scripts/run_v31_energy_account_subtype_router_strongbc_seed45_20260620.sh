#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

"$PY" scripts/58_v31_split_protocol_run.py \
  --out-dir reports/v31_energy_account_subtype_router_strongbc_seed45_h082_20260620 \
  --sensor-cfg configs/sensors/windblown_sensors_v31_specialist_subtype.yaml \
  --seed 45 \
  --budget 1.05 \
  --startup-peak-budget 1.30 \
  --truth-steps 60000 \
  --freq-s 10800 \
  --split-ratios 0.35 0.50 0.075 0.075 \
  --event-coverage 0.55 \
  --min-duration 12 \
  --max-duration 36 \
  --min-gap 2 \
  --lead-steps 6 \
  --wind-margin-ms 1.2 \
  --cred-hysteresis-on 0.6 \
  --cred-hysteresis-off 0.3 \
  --flux-wind-exponent 4.0 \
  --event-microstructure-sigma 0.70 \
  --event-microstructure-alpha 0.20 \
  --event-microstructure-diameter-scale 0.18 \
  --event-microstructure-velocity-scale 1.7 \
  --event-particle-microstructure-correlation 0.0 \
  --event-subtypes-enabled \
  --event-subtype-particle-prob 0.36 \
  --event-subtype-flux-prob 0.40 \
  --event-subtype-thermal-prob 0.24 \
  --event-subtype-particle-flux-multiplier 0.70 \
  --event-subtype-flux-multiplier 3.4 \
  --event-subtype-thermal-flux-multiplier 0.35 \
  --event-subtype-particle-diameter-shift-mm 0.14 \
  --event-subtype-particle-velocity-boost-ms 1.7 \
  --event-subtype-flux-diameter-shift-mm -0.06 \
  --event-subtype-flux-velocity-boost-ms 0.9 \
  --event-subtype-thermal-surface-drop-c 2.2 \
  --event-subtype-particle-humidity-boost-pct 24.0 \
  --event-subtype-flux-wind-boost-ms 3.2 \
  --event-subtype-thermal-air-temp-drop-c 2.4 \
  --oracle-rollout-steps 2400 \
  --oracle-rollouts-per-policy 6 \
  --oracle-epochs 18 \
  --oracle-batch-size 512 \
  --oracle-device auto \
  --oracle-inference-device cpu \
  --total-timesteps 60000 \
  --n-steps 512 \
  --batch-size 128 \
  --n-epochs 8 \
  --ent-coef 0.0002 \
  --awbc-coef 1.5 \
  --awbc-label-stride 1 \
  --bc-pretrain-steps 40000 \
  --bc-pretrain-epochs 10 \
  --bc-pretrain-batch-size 256 \
  --bc-pretrain-loss-coef 1.0 \
  --subtype-aux-coef 0.75 \
  --subtype-aux-classes 4 \
  --subtype-aux-lookahead-steps 0 \
  --subtype-router \
  --subtype-router-min-confidence 0.86 \
  --subtype-router-low-confidence-action 13 \
  --awbc-teacher-mode subtype_auto \
  --awbc-teacher-subtype-calm-sensors shielded_thermo_hygro \
  --awbc-teacher-subtype-particle-sensors shielded_thermo_hygro laser_disdrometer \
  --awbc-teacher-subtype-flux-sensors shielded_thermo_hygro fc4_flux \
  --awbc-teacher-subtype-thermal-sensors shielded_thermo_hygro surface_temp_ir \
  --prior-kl-coef 0.0 \
  --greedy-lookahead-steps 4 \
  --event-start-prob 0.65 \
  --event-aware-critic \
  --event-gated-actor \
  --soc-aux-horizon 48 \
  --soc-aux-coef 0.06 \
  --train-episode-len 512 \
  --use-candidate-prior \
  --candidate-prior-scale 0.5 \
  --candidate-prior-steps 512 \
  --candidate-prior-rollouts 4 \
  --static-selection-steps 512 \
  --static-selection-rollouts 4 \
  --eval-steps 512 \
  --eval-rollouts 8 \
  --eval-start-selection event_transport_rich \
  --eval-event-fraction 0.70 \
  --eval-selection-stride 64 \
  --lambda-warmup-abort 1.0 \
  --lambda-switch 0.002 \
  --event-reward-multiplier 1.5 \
  --energy-account \
  --energy-capacity 80.0 \
  --initial-energy 80.0 \
  --harvest-per-step 0.52 \
  --reserve-energy 10.0 \
  --lambda-energy-deficit 2.0 \
  --soc-soft-penalty-buffer 15.0 \
  --lambda-soc-soft-penalty 0.06 \
  --lambda-duty-balance 0.0 \
  --duty-score-feedback 0.0 \
  --no-duty-hard-guard \
  --min-dwell-steps 8 \
  --include-observable-regime-belief \
  --regime-belief-lookback 8 \
  --target-weights 0.03 1.0 0.2 0.01 0.01 0.0 60.0 60.0 60.0 \
  --target-scales 5.0 5.0 5.0 1.0 1.0 100.0 0.0001 0.2 5.0 \
  --disable-coverage-groups \
  --max-active 3 \
  --skip-rollout-evaluation
