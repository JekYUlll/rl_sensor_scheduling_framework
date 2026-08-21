#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-8}"

RUN_PREFIX="${RUN_PREFIX:-v31_metpair_strongclaim}"
DATE_TAG="${DATE_TAG:-20260620}"
BUDGET="${BUDGET:-0.75}"
BUDGET_LABEL="${BUDGET_LABEL:-h075}"
DEVICE="${DEVICE:-cuda}"
ORACLE_DEVICE="${ORACLE_DEVICE:-auto}"
ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-cpu}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(46 47)
fi

run_seed() {
  local seed="$1"
  local out_dir="reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${DATE_TAG}"
  mkdir -p "$out_dir"

  echo "[metpair] seed=${seed} out_dir=${out_dir}"

  if [[ ! -f "${out_dir}/custom_ppo.pt" ]]; then
    "$PY" scripts/58_v31_split_protocol_run.py \
      --out-dir "$out_dir" \
      --sensor-cfg configs/sensors/windblown_sensors_v31_met_specialist_pair.yaml \
      --seed "$seed" \
      --budget "$BUDGET" \
      --startup-peak-budget 0.95 \
      --truth-steps 70000 \
      --freq-s 3600 \
      --split-ratios 0.35 0.50 0.075 0.075 \
      --event-coverage 0.55 \
      --min-duration 28 \
      --max-duration 80 \
      --min-gap 8 \
      --lead-steps 8 \
      --wind-margin-ms 1.4 \
      --cred-hysteresis-on 0.6 \
      --cred-hysteresis-off 0.3 \
      --flux-wind-exponent 3.2 \
      --event-microstructure-sigma 0.0 \
      --event-microstructure-alpha 0.18 \
      --event-microstructure-diameter-scale 0.0 \
      --event-microstructure-velocity-scale 0.0 \
      --event-particle-microstructure-correlation 1.0 \
      --event-subtypes-enabled \
      --event-subtype-particle-prob 0.45 \
      --event-subtype-flux-prob 0.45 \
      --event-subtype-thermal-prob 0.10 \
      --event-subtype-particle-flux-multiplier 0.45 \
      --event-subtype-flux-multiplier 5.0 \
      --event-subtype-thermal-flux-multiplier 0.35 \
      --event-subtype-particle-diameter-shift-mm 0.15 \
      --event-subtype-particle-velocity-boost-ms 2.2 \
      --event-subtype-flux-diameter-shift-mm -0.08 \
      --event-subtype-flux-velocity-boost-ms 1.0 \
      --event-subtype-thermal-surface-drop-c 3.0 \
      --event-subtype-particle-humidity-boost-pct 0.0 \
      --event-subtype-flux-wind-boost-ms 0.0 \
      --event-subtype-thermal-air-temp-drop-c 0.0 \
      --event-subtype-latent-alpha 0.25 \
      --event-subtype-particle-latent-diameter-scale-mm 0.18 \
      --event-subtype-particle-latent-velocity-scale-ms 3.0 \
      --event-subtype-flux-latent-sigma 1.6 \
      --event-subtype-thermal-latent-surface-scale-c 3.0 \
      --event-subtype-latent-target-lag-steps 4 \
      --event-subtype-context-lead-steps 8 \
      --event-subtype-context-noise-std 0.03 \
      --oracle-rollout-steps 4096 \
      --oracle-rollouts-per-policy 6 \
      --oracle-epochs 18 \
      --oracle-batch-size 512 \
      --oracle-loss-clip 20 \
      --oracle-candidate-mask-repeat 1 \
      --oracle-candidate-mask-limit 0 \
      --oracle-subtype-teacher-repeat 8 \
      --oracle-subtype-teacher-lookahead-steps 8 \
      --oracle-subtype-teacher-calm-sensors met_station_core shielded_thermo_hygro \
      --oracle-subtype-teacher-particle-sensors met_station_core laser_disdrometer \
      --oracle-subtype-teacher-flux-sensors met_station_core fc4_flux \
      --oracle-subtype-teacher-thermal-sensors met_station_core surface_temp_ir \
      --oracle-device "$ORACLE_DEVICE" \
      --oracle-inference-device "$ORACLE_INFERENCE_DEVICE" \
      --total-timesteps 120000 \
      --n-steps 1024 \
      --batch-size 128 \
      --n-epochs 8 \
      --ent-coef 0.02 \
      --awbc-coef 0.45 \
      --awbc-label-stride 2 \
      --bc-pretrain-steps 3500 \
      --bc-pretrain-epochs 8 \
      --bc-pretrain-batch-size 256 \
      --bc-pretrain-loss-coef 1.2 \
      --subtype-aux-coef 1.0 \
      --subtype-aux-classes 4 \
      --subtype-aux-lookahead-steps 8 \
      --subtype-router \
      --subtype-router-min-confidence 0.0 \
      --subtype-router-low-confidence-action -1 \
      --awbc-teacher-mode subtype_auto \
      --awbc-teacher-event-lookahead-steps 8 \
      --awbc-teacher-subtype-calm-sensors met_station_core shielded_thermo_hygro \
      --awbc-teacher-subtype-particle-sensors met_station_core laser_disdrometer \
      --awbc-teacher-subtype-flux-sensors met_station_core fc4_flux \
      --awbc-teacher-subtype-thermal-sensors met_station_core surface_temp_ir \
      --awbc-teacher-dwell-steps 8 \
      --prior-kl-coef 0.0 \
      --greedy-lookahead-steps 4 \
      --event-start-prob 0.85 \
      --event-aware-critic \
      --no-event-gated-actor \
      --soc-aux-horizon 0 \
      --soc-aux-coef 0.0 \
      --train-episode-len 512 \
      --no-use-candidate-prior \
      --candidate-prior-scale 2.0 \
      --candidate-prior-steps 512 \
      --candidate-prior-rollouts 4 \
      --static-selection-steps 512 \
      --static-selection-rollouts 4 \
      --eval-steps 512 \
      --eval-rollouts 8 \
      --eval-start-selection event_transport_rich \
      --eval-event-fraction 0.75 \
      --eval-selection-stride 64 \
      --lambda-warmup-abort 1.0 \
      --lambda-switch 0.002 \
      --event-reward-multiplier 1.0 \
      --lambda-duty-balance 0.0 \
      --duty-score-feedback 0.0 \
      --no-duty-hard-guard \
      --no-primary-eval-duty-guard \
      --min-dwell-steps 8 \
      --target-weights 0.02 0.05 0.05 0.0 0.0 0.0 25.0 12.0 12.0 \
      --subtype-loss-weighting \
      --subtype-particle-target-weights 0.0 0.0 0.05 0.0 0.0 0.0 1.0 20.0 20.0 \
      --subtype-flux-target-weights 0.0 0.0 0.05 0.0 0.0 0.0 30.0 1.0 1.0 \
      --subtype-thermal-target-weights 0.05 15.0 0.05 0.0 0.0 0.0 0.2 0.2 0.2 \
      --disable-coverage-groups \
      --max-active 2 \
      --device "$DEVICE" \
      2>&1 | tee "${out_dir}/run_train_eval.log"
  else
    echo "[metpair] seed=${seed} training artifact exists; skipping train"
  fi

  if [[ ! -f "${out_dir}/v2_custom_ppo_metrics.csv" ]]; then
    "$PY" scripts/64_v31_eval_saved_run_operational_baselines.py \
      --source-run-dir "$out_dir" \
      --out-dir "$out_dir" \
      --device "$DEVICE" \
      --oracle-device "$EVAL_ORACLE_DEVICE" \
      2>&1 | tee "${out_dir}/eval_standard.log"
  fi

  if [[ ! -f "${out_dir}/eval_router_conf08/v2_custom_ppo_metrics.csv" ]]; then
    "$PY" scripts/64_v31_eval_saved_run_operational_baselines.py \
      --source-run-dir "$out_dir" \
      --out-dir "${out_dir}/eval_router_conf08" \
      --device "$DEVICE" \
      --oracle-device "$EVAL_ORACLE_DEVICE" \
      --subtype-router \
      --subtype-router-min-confidence 0.8 \
      2>&1 | tee "${out_dir}/eval_router_conf08.log"
  fi

  if [[ ! -f "${out_dir}/replay_gate_explicit_static_noguard/split_replay_gate_summary.json" ]]; then
    "$PY" scripts/70_v31_split_replay_gate.py \
      --source-run-dir "$out_dir" \
      --out-dir "${out_dir}/replay_gate_explicit_static_noguard" \
      --oracle-device "$EVAL_ORACLE_DEVICE" \
      --replay-family subtype_explicit \
      --explicit-policy-name split_metpair_subtype_explicit \
      --explicit-calm-sensors met_station_core shielded_thermo_hygro \
      --explicit-particle-sensors met_station_core laser_disdrometer \
      --explicit-flux-sensors met_station_core fc4_flux \
      --explicit-thermal-sensors met_station_core surface_temp_ir \
      --lead-steps 0 2 4 8 10 \
      --dwell-steps 6 12 24 \
      --subtype-top-size-cap 2 \
      --static-reference-duty-guard off \
      --enforce-static-candidate-reference \
      --min-margin-abs 0.005 \
      --min-margin-rel 0.01 \
      2>&1 | tee "${out_dir}/replay_gate_explicit_static_noguard.log"
  fi

  if [[ ! -f "${out_dir}/behavior_audit_v2/behavior_complexity_summary.json" ]]; then
    "$PY" scripts/71_v31_behavior_complexity_audit.py \
      --out-dir "${out_dir}/behavior_audit_v2" \
      "${out_dir}/eval_router_conf08/rollout_custom_ppo.npz" \
      "${out_dir}/eval_router_conf08/rollout_validation_selected_static.npz" \
      "${out_dir}/rollout_custom_ppo.npz" \
      2>&1 | tee "${out_dir}/behavior_audit_v2.log"
  fi
}

for seed in "${SEEDS[@]}"; do
  run_seed "$seed"
done
