#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

"$PY" scripts/58_v31_split_protocol_run.py \
  --out-dir reports/v31_subtype_snowlatent_tight_candidateoracle_tcn_gate_seed45_h082_20260620 \
  --sensor-cfg configs/sensors/windblown_sensors_v31_specialist_subtype_tight.yaml \
  --seed 45 \
  --budget 0.95 \
  --startup-peak-budget 1.15 \
  --truth-steps 60000 \
  --freq-s 10800 \
  --split-ratios 0.35 0.50 0.075 0.075 \
  --event-coverage 0.60 \
  --min-duration 12 \
  --max-duration 36 \
  --min-gap 2 \
  --lead-steps 6 \
  --wind-margin-ms 1.2 \
  --cred-hysteresis-on 0.6 \
  --cred-hysteresis-off 0.3 \
  --flux-wind-exponent 4.5 \
  --event-microstructure-sigma 0.25 \
  --event-microstructure-alpha 0.20 \
  --event-microstructure-diameter-scale 0.03 \
  --event-microstructure-velocity-scale 0.4 \
  --event-particle-microstructure-correlation 0.0 \
  --event-subtypes-enabled \
  --event-subtype-particle-prob 0.45 \
  --event-subtype-flux-prob 0.45 \
  --event-subtype-thermal-prob 0.10 \
  --event-subtype-particle-flux-multiplier 0.08 \
  --event-subtype-flux-multiplier 10.0 \
  --event-subtype-thermal-flux-multiplier 0.08 \
  --event-subtype-particle-diameter-shift-mm 0.14 \
  --event-subtype-particle-velocity-boost-ms 1.8 \
  --event-subtype-flux-diameter-shift-mm 0.00 \
  --event-subtype-flux-velocity-boost-ms 0.00 \
  --event-subtype-thermal-surface-drop-c 2.0 \
  --event-subtype-particle-humidity-boost-pct 24.0 \
  --event-subtype-flux-wind-boost-ms 0.4 \
  --event-subtype-thermal-air-temp-drop-c 1.0 \
  --event-subtype-latent-alpha 0.12 \
  --event-subtype-particle-latent-diameter-scale-mm 0.13 \
  --event-subtype-particle-latent-velocity-scale-ms 3.0 \
  --event-subtype-flux-latent-sigma 2.4 \
  --event-subtype-thermal-latent-surface-scale-c 2.0 \
  --oracle-rollout-steps 3600 \
  --oracle-rollouts-per-policy 6 \
  --oracle-epochs 12 \
  --oracle-batch-size 512 \
  --oracle-candidate-mask-repeat 1 \
  --oracle-candidate-mask-limit 0 \
  --oracle-device auto \
  --oracle-inference-device cpu \
  --total-timesteps 0 \
  --n-steps 384 \
  --batch-size 128 \
  --n-epochs 4 \
  --ent-coef 0.001 \
  --awbc-coef 0.0 \
  --awbc-label-stride 8 \
  --bc-pretrain-steps 0 \
  --subtype-aux-coef 0.0 \
  --no-subtype-router \
  --awbc-teacher-mode subtype_auto \
  --awbc-teacher-subtype-calm-sensors shielded_thermo_hygro \
  --awbc-teacher-subtype-particle-sensors shielded_thermo_hygro laser_disdrometer \
  --awbc-teacher-subtype-flux-sensors shielded_thermo_hygro fc4_flux \
  --awbc-teacher-subtype-thermal-sensors shielded_thermo_hygro surface_temp_ir \
  --prior-kl-coef 0.0 \
  --greedy-lookahead-steps 4 \
  --event-start-prob 0.70 \
  --event-aware-critic \
  --event-gated-actor \
  --soc-aux-horizon 0 \
  --soc-aux-coef 0.0 \
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
  --eval-event-fraction 0.80 \
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
  --target-weights 0.03 2.0 0.10 0.01 0.01 0.0 500.0 180.0 180.0 \
  --target-scales 5.0 5.0 5.0 1.0 1.0 100.0 0.0001 0.2 5.0 \
  --disable-coverage-groups \
  --max-active 2 \
  --skip-rollout-evaluation
