#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-8}"

RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context}"
DATE_TAG="${DATE_TAG:-20260620}"
BUDGET="${BUDGET:-0.75}"
BUDGET_LABEL="${BUDGET_LABEL:-h075ctx}"
CONTROL_SOURCE_RUN_PREFIX="${CONTROL_SOURCE_RUN_PREFIX:-}"
CONTROL_SOURCE_BUDGET_LABEL="${CONTROL_SOURCE_BUDGET_LABEL:-h075ctxolscbal2}"
CONTROL_SOURCE_DATE_TAG="${CONTROL_SOURCE_DATE_TAG:-20260621}"
DEVICE="${DEVICE:-cuda}"
ORACLE_DEVICE="${ORACLE_DEVICE:-auto}"
ORACLE_INFERENCE_DEVICE="${ORACLE_INFERENCE_DEVICE:-cpu}"
EVAL_ORACLE_DEVICE="${EVAL_ORACLE_DEVICE:-cpu}"
EVENT_COVERAGE="${EVENT_COVERAGE:-0.62}"
PARTICLE_PROB="${PARTICLE_PROB:-0.34}"
FLUX_PROB="${FLUX_PROB:-0.33}"
THERMAL_PROB="${THERMAL_PROB:-0.33}"
PARTICLE_FLUX_MULTIPLIER="${PARTICLE_FLUX_MULTIPLIER:-0.35}"
FLUX_MULTIPLIER="${FLUX_MULTIPLIER:-6.0}"
THERMAL_FLUX_MULTIPLIER="${THERMAL_FLUX_MULTIPLIER:-0.25}"
THERMAL_SURFACE_DROP_C="${THERMAL_SURFACE_DROP_C:-4.0}"
THERMAL_AIR_TEMP_DROP_C="${THERMAL_AIR_TEMP_DROP_C:-1.0}"
SUBTYPE_LATENT_ALPHA="${SUBTYPE_LATENT_ALPHA:-0.25}"
PARTICLE_LATENT_DIAMETER_SCALE_MM="${PARTICLE_LATENT_DIAMETER_SCALE_MM:-0.18}"
PARTICLE_LATENT_VELOCITY_SCALE_MS="${PARTICLE_LATENT_VELOCITY_SCALE_MS:-3.0}"
FLUX_LATENT_SIGMA="${FLUX_LATENT_SIGMA:-1.8}"
FLUX_LATENT_LINEAR_SCALE="${FLUX_LATENT_LINEAR_SCALE:-0.0}"
FLUX_LATENT_LINEAR_OFFSET="${FLUX_LATENT_LINEAR_OFFSET:-1.5}"
FLUX_LATENT_LINEAR_CLIP="${FLUX_LATENT_LINEAR_CLIP:-4.0}"
THERMAL_LATENT_SURFACE_SCALE_C="${THERMAL_LATENT_SURFACE_SCALE_C:-4.0}"
SUBTYPE_CONTEXT_NOISE_STD="${SUBTYPE_CONTEXT_NOISE_STD:-0.01}"
SUBTYPE_LATENT_TARGET_LAG_STEPS="${SUBTYPE_LATENT_TARGET_LAG_STEPS:-4}"
SUBTYPE_CONTEXT_LEAD_STEPS="${SUBTYPE_CONTEXT_LEAD_STEPS:-8}"
BLOWING_SNOW_LEAD_STEPS="${BLOWING_SNOW_LEAD_STEPS:-8}"
ORACLE_SUBTYPE_TEACHER_REPEAT="${ORACLE_SUBTYPE_TEACHER_REPEAT:-12}"
ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS="${ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS:-8}"
AWBC_COEF="${AWBC_COEF:-0.8}"
AWBC_LABEL_STRIDE="${AWBC_LABEL_STRIDE:-2}"
AWBC_TEACHER_MODE="${AWBC_TEACHER_MODE:-subtype_auto}"
AWBC_TEACHER_AUTO_SCORE_MODE="${AWBC_TEACHER_AUTO_SCORE_MODE:-raw}"
AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS="${AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS:-8}"
AWBC_TEACHER_DWELL_STEPS="${AWBC_TEACHER_DWELL_STEPS:-8}"
BC_PRETRAIN_LOSS_COEF="${BC_PRETRAIN_LOSS_COEF:-2.0}"
BC_PRETRAIN_STEPS="${BC_PRETRAIN_STEPS:-3500}"
BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS:-8}"
BC_PRETRAIN_BATCH_SIZE="${BC_PRETRAIN_BATCH_SIZE:-256}"
SUBTYPE_AUX_COEF="${SUBTYPE_AUX_COEF:-2.0}"
SUBTYPE_AUX_LOOKAHEAD_STEPS="${SUBTYPE_AUX_LOOKAHEAD_STEPS:-8}"
SUBTYPE_ACTION_CE_COEF="${SUBTYPE_ACTION_CE_COEF:-0.0}"
SUBTYPE_ACTION_MARGIN_COEF="${SUBTYPE_ACTION_MARGIN_COEF:-0.0}"
SUBTYPE_ACTION_MARGIN="${SUBTYPE_ACTION_MARGIN:-0.5}"
SUBTYPE_ROUTER_ENABLED="${SUBTYPE_ROUTER_ENABLED:-1}"
SUBTYPE_ROUTER_MIN_CONFIDENCE="${SUBTYPE_ROUTER_MIN_CONFIDENCE:-0.0}"
EVAL_SUBTYPE_ROUTER_MIN_CONFIDENCE="${EVAL_SUBTYPE_ROUTER_MIN_CONFIDENCE:-0.8}"
INCLUDE_OBSERVABLE_REGIME_BELIEF="${INCLUDE_OBSERVABLE_REGIME_BELIEF:-0}"
REGIME_BELIEF_LOOKBACK="${REGIME_BELIEF_LOOKBACK:-6}"
EVENT_AWARE_CRITIC="${EVENT_AWARE_CRITIC:-1}"
EVENT_GATED_ACTOR="${EVENT_GATED_ACTOR:-0}"
CONTEXT_ENCODER="${CONTEXT_ENCODER:-0}"
CONTEXT_FEATURE_DIM="${CONTEXT_FEATURE_DIM:-0}"
CONTEXT_HIDDEN_DIM="${CONTEXT_HIDDEN_DIM:-64}"
CONTEXT_FUSION_MODE="${CONTEXT_FUSION_MODE:-concat}"
CONTEXT_LAYER_NORM="${CONTEXT_LAYER_NORM:-0}"
INCLUDE_EVENT_FLAG_IN_STATE="${INCLUDE_EVENT_FLAG_IN_STATE:-1}"
INCLUDE_ALERT_CONTEXT_FEATURES="${INCLUDE_ALERT_CONTEXT_FEATURES:-0}"
ALERT_CONTEXT_THRESHOLD="${ALERT_CONTEXT_THRESHOLD:-0.5}"
ALERT_CONTEXT_TREND_LOOKBACK="${ALERT_CONTEXT_TREND_LOOKBACK:-6}"
SKIP_ROUTER_EVAL="${SKIP_ROUTER_EVAL:-0}"
SKIP_REPLAY_GATE="${SKIP_REPLAY_GATE:-0}"
SKIP_BEHAVIOR_AUDIT="${SKIP_BEHAVIOR_AUDIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
N_STEPS="${N_STEPS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_EPOCHS="${N_EPOCHS:-8}"
ENT_COEF="${ENT_COEF:-0.01}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
TRUTH_STEPS="${TRUTH_STEPS:-70000}"
ORACLE_ROLLOUT_STEPS="${ORACLE_ROLLOUT_STEPS:-4096}"
ORACLE_ROLLOUTS_PER_POLICY="${ORACLE_ROLLOUTS_PER_POLICY:-6}"
ORACLE_EPOCHS="${ORACLE_EPOCHS:-18}"
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-512}"
ORACLE_LOSS_CLIP="${ORACLE_LOSS_CLIP:-20}"
STATIC_SELECTION_STEPS="${STATIC_SELECTION_STEPS:-512}"
EVAL_STEPS="${EVAL_STEPS:-512}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-8}"
MIN_DWELL_STEPS="${MIN_DWELL_STEPS:-8}"
GREEDY_LOOKAHEAD_STEPS="${GREEDY_LOOKAHEAD_STEPS:-4}"
REPLAY_LEAD_STEPS="${REPLAY_LEAD_STEPS:-0 2 4 8 10}"
REPLAY_DWELL_STEPS="${REPLAY_DWELL_STEPS:-6 12 24}"
LAMBDA_SWITCH="${LAMBDA_SWITCH:-0.002}"
LAMBDA_DUTY_BALANCE="${LAMBDA_DUTY_BALANCE:-0.0}"
DUTY_BALANCE_LOW="${DUTY_BALANCE_LOW:-0.05}"
DUTY_BALANCE_HIGH="${DUTY_BALANCE_HIGH:-0.95}"
DUTY_BALANCE_GRACE_STEPS="${DUTY_BALANCE_GRACE_STEPS:-64}"
DUTY_SCORE_FEEDBACK="${DUTY_SCORE_FEEDBACK:-0.0}"
DUTY_SCORE_TARGET="${DUTY_SCORE_TARGET:-0.40}"
DUTY_HARD_GUARD="${DUTY_HARD_GUARD:-0}"
DUTY_HARD_LOW="${DUTY_HARD_LOW:-0.08}"
DUTY_HARD_HIGH="${DUTY_HARD_HIGH:-0.92}"
DUTY_HARD_SCORE="${DUTY_HARD_SCORE:-8.0}"
EVENT_SUBTYPE_PARTICLE_REWARD_MULTIPLIER="${EVENT_SUBTYPE_PARTICLE_REWARD_MULTIPLIER:-1.0}"
EVENT_SUBTYPE_FLUX_REWARD_MULTIPLIER="${EVENT_SUBTYPE_FLUX_REWARD_MULTIPLIER:-1.0}"
EVENT_SUBTYPE_THERMAL_REWARD_MULTIPLIER="${EVENT_SUBTYPE_THERMAL_REWARD_MULTIPLIER:-1.0}"
REWARD_LOSS_NORMALIZATION="${REWARD_LOSS_NORMALIZATION:-none}"
REWARD_PROXY_MODE="${REWARD_PROXY_MODE:-forecast}"
STATIC_SELECTION_SCORE="${STATIC_SELECTION_SCORE:-oracle_loss_mean}"
METRICS_SORT_SCORE="${METRICS_SORT_SCORE:-oracle_loss_mean}"
MACRO_SCORE_COLUMN="${MACRO_SCORE_COLUMN:-oracle_loss_macro_subtype_event}"
TARGET_WEIGHTS="${TARGET_WEIGHTS:-0.02 0.05 0.05 0.0 0.0 0.0 25.0 12.0 12.0}"
SUBTYPE_PARTICLE_TARGET_WEIGHTS="${SUBTYPE_PARTICLE_TARGET_WEIGHTS:-0.0 0.0 0.05 0.0 0.0 0.0 1.0 20.0 20.0}"
SUBTYPE_FLUX_TARGET_WEIGHTS="${SUBTYPE_FLUX_TARGET_WEIGHTS:-0.0 0.0 0.05 0.0 0.0 0.0 30.0 1.0 1.0}"
SUBTYPE_THERMAL_TARGET_WEIGHTS="${SUBTYPE_THERMAL_TARGET_WEIGHTS:-0.05 15.0 0.05 0.0 0.0 0.0 0.2 0.2 0.2}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(41 42)
fi

run_seed() {
  local seed="$1"
  local out_dir="reports/${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${DATE_TAG}"
  mkdir -p "$out_dir"
  local control_source_args=()
  if [[ -n "$CONTROL_SOURCE_RUN_PREFIX" ]]; then
    local source_dir="reports/${CONTROL_SOURCE_RUN_PREFIX}_seed${seed}_${CONTROL_SOURCE_BUDGET_LABEL}_${CONTROL_SOURCE_DATE_TAG}"
    if [[ ! -d "$source_dir" ]]; then
      echo "Missing control source run: ${source_dir}" >&2
      return 2
    fi
    control_source_args+=(--truth-csv "${source_dir}/truth_v31_split.csv")
    control_source_args+=(--control-source-run-dir "$source_dir")
  fi
  local duty_guard_args=()
  if [[ "$DUTY_HARD_GUARD" == "1" || "$DUTY_HARD_GUARD" == "true" || "$DUTY_HARD_GUARD" == "yes" ]]; then
    duty_guard_args+=(--duty-hard-guard)
  else
    duty_guard_args+=(--no-duty-hard-guard)
  fi
  local regime_belief_args=()
  if [[ "$INCLUDE_OBSERVABLE_REGIME_BELIEF" == "1" || "$INCLUDE_OBSERVABLE_REGIME_BELIEF" == "true" || "$INCLUDE_OBSERVABLE_REGIME_BELIEF" == "yes" ]]; then
    regime_belief_args+=(--include-observable-regime-belief)
  fi
  local event_gated_actor_args=()
  if [[ "$EVENT_GATED_ACTOR" == "1" || "$EVENT_GATED_ACTOR" == "true" || "$EVENT_GATED_ACTOR" == "yes" ]]; then
    event_gated_actor_args+=(--event-gated-actor)
  else
    event_gated_actor_args+=(--no-event-gated-actor)
  fi
  local event_aware_critic_args=()
  if [[ "$EVENT_AWARE_CRITIC" == "1" || "$EVENT_AWARE_CRITIC" == "true" || "$EVENT_AWARE_CRITIC" == "yes" ]]; then
    event_aware_critic_args+=(--event-aware-critic)
  else
    event_aware_critic_args+=(--no-event-aware-critic)
  fi
  local context_encoder_args=()
  if [[ "$CONTEXT_ENCODER" == "1" || "$CONTEXT_ENCODER" == "true" || "$CONTEXT_ENCODER" == "yes" ]]; then
    context_encoder_args+=(--context-encoder)
  else
    context_encoder_args+=(--no-context-encoder)
  fi
  local context_layer_norm_args=()
  if [[ "$CONTEXT_LAYER_NORM" == "1" || "$CONTEXT_LAYER_NORM" == "true" || "$CONTEXT_LAYER_NORM" == "yes" ]]; then
    context_layer_norm_args+=(--context-layer-norm)
  else
    context_layer_norm_args+=(--no-context-layer-norm)
  fi
  local event_flag_state_args=()
  if [[ "$INCLUDE_EVENT_FLAG_IN_STATE" == "1" || "$INCLUDE_EVENT_FLAG_IN_STATE" == "true" || "$INCLUDE_EVENT_FLAG_IN_STATE" == "yes" ]]; then
    event_flag_state_args+=(--include-event-flag-in-state)
  else
    event_flag_state_args+=(--no-include-event-flag-in-state)
  fi
  local alert_context_args=()
  if [[ "$INCLUDE_ALERT_CONTEXT_FEATURES" == "1" || "$INCLUDE_ALERT_CONTEXT_FEATURES" == "true" || "$INCLUDE_ALERT_CONTEXT_FEATURES" == "yes" ]]; then
    alert_context_args+=(--include-alert-context-features)
  fi
  local subtype_router_args=()
  if [[ "$SUBTYPE_ROUTER_ENABLED" == "1" || "$SUBTYPE_ROUTER_ENABLED" == "true" || "$SUBTYPE_ROUTER_ENABLED" == "yes" ]]; then
    subtype_router_args+=(--subtype-router)
  else
    subtype_router_args+=(--no-subtype-router)
  fi
  local dry_run_args=()
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
    dry_run_args+=(--dry-run)
  fi
  local preflight_args=()
  if [[ "$PREFLIGHT_ONLY" == "1" || "$PREFLIGHT_ONLY" == "true" || "$PREFLIGHT_ONLY" == "yes" ]]; then
    preflight_args+=(--validate-control-source-only)
  fi

  echo "[metpair-backbone-context] seed=${seed} out_dir=${out_dir}"

  if [[ ! -f "${out_dir}/custom_ppo.pt" ]]; then
    "$PY" scripts/58_v31_split_protocol_run.py \
      --out-dir "$out_dir" \
      "${control_source_args[@]}" \
      --sensor-cfg configs/sensors/windblown_sensors_v31_met_specialist_pair.yaml \
      --seed "$seed" \
      --budget "$BUDGET" \
      --startup-peak-budget 0.95 \
      --truth-steps "$TRUTH_STEPS" \
      --freq-s 3600 \
      --split-ratios 0.35 0.50 0.075 0.075 \
      --event-coverage "$EVENT_COVERAGE" \
      --min-duration 28 \
      --max-duration 80 \
      --min-gap 8 \
      --lead-steps "$BLOWING_SNOW_LEAD_STEPS" \
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
      --event-subtype-particle-prob "$PARTICLE_PROB" \
      --event-subtype-flux-prob "$FLUX_PROB" \
      --event-subtype-thermal-prob "$THERMAL_PROB" \
      --event-subtype-particle-flux-multiplier "$PARTICLE_FLUX_MULTIPLIER" \
      --event-subtype-flux-multiplier "$FLUX_MULTIPLIER" \
      --event-subtype-thermal-flux-multiplier "$THERMAL_FLUX_MULTIPLIER" \
      --event-subtype-particle-diameter-shift-mm 0.15 \
      --event-subtype-particle-velocity-boost-ms 2.2 \
      --event-subtype-flux-diameter-shift-mm -0.08 \
      --event-subtype-flux-velocity-boost-ms 1.0 \
      --event-subtype-thermal-surface-drop-c "$THERMAL_SURFACE_DROP_C" \
      --event-subtype-particle-humidity-boost-pct 0.0 \
      --event-subtype-flux-wind-boost-ms 0.0 \
      --event-subtype-thermal-air-temp-drop-c "$THERMAL_AIR_TEMP_DROP_C" \
      --event-subtype-latent-alpha "$SUBTYPE_LATENT_ALPHA" \
      --event-subtype-particle-latent-diameter-scale-mm "$PARTICLE_LATENT_DIAMETER_SCALE_MM" \
      --event-subtype-particle-latent-velocity-scale-ms "$PARTICLE_LATENT_VELOCITY_SCALE_MS" \
      --event-subtype-flux-latent-sigma "$FLUX_LATENT_SIGMA" \
      --event-subtype-flux-latent-linear-scale "$FLUX_LATENT_LINEAR_SCALE" \
      --event-subtype-flux-latent-linear-offset "$FLUX_LATENT_LINEAR_OFFSET" \
      --event-subtype-flux-latent-linear-clip "$FLUX_LATENT_LINEAR_CLIP" \
      --event-subtype-thermal-latent-surface-scale-c "$THERMAL_LATENT_SURFACE_SCALE_C" \
      --event-subtype-latent-target-lag-steps "$SUBTYPE_LATENT_TARGET_LAG_STEPS" \
      --event-subtype-context-lead-steps "$SUBTYPE_CONTEXT_LEAD_STEPS" \
      --event-subtype-context-noise-std "$SUBTYPE_CONTEXT_NOISE_STD" \
      --oracle-rollout-steps "$ORACLE_ROLLOUT_STEPS" \
      --oracle-rollouts-per-policy "$ORACLE_ROLLOUTS_PER_POLICY" \
      --oracle-epochs "$ORACLE_EPOCHS" \
      --oracle-batch-size "$ORACLE_BATCH_SIZE" \
      --oracle-loss-clip "$ORACLE_LOSS_CLIP" \
      --oracle-candidate-mask-repeat 1 \
      --oracle-candidate-mask-limit 0 \
      --oracle-subtype-teacher-repeat "$ORACLE_SUBTYPE_TEACHER_REPEAT" \
      --oracle-subtype-teacher-lookahead-steps "$ORACLE_SUBTYPE_TEACHER_LOOKAHEAD_STEPS" \
      --oracle-subtype-teacher-calm-sensors met_station_core shielded_thermo_hygro \
      --oracle-subtype-teacher-particle-sensors met_station_core laser_disdrometer \
      --oracle-subtype-teacher-flux-sensors met_station_core fc4_flux \
      --oracle-subtype-teacher-thermal-sensors met_station_core surface_temp_ir \
      --oracle-device "$ORACLE_DEVICE" \
      --oracle-inference-device "$ORACLE_INFERENCE_DEVICE" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --n-steps "$N_STEPS" \
      --batch-size "$BATCH_SIZE" \
      --n-epochs "$N_EPOCHS" \
      --ent-coef "$ENT_COEF" \
      --awbc-coef "$AWBC_COEF" \
      --awbc-label-stride "$AWBC_LABEL_STRIDE" \
      --bc-pretrain-steps "$BC_PRETRAIN_STEPS" \
      --bc-pretrain-epochs "$BC_PRETRAIN_EPOCHS" \
      --bc-pretrain-batch-size "$BC_PRETRAIN_BATCH_SIZE" \
      --bc-pretrain-loss-coef "$BC_PRETRAIN_LOSS_COEF" \
      --subtype-aux-coef "$SUBTYPE_AUX_COEF" \
      --subtype-aux-classes 4 \
      --subtype-aux-lookahead-steps "$SUBTYPE_AUX_LOOKAHEAD_STEPS" \
      --subtype-action-ce-coef "$SUBTYPE_ACTION_CE_COEF" \
      --subtype-action-margin-coef "$SUBTYPE_ACTION_MARGIN_COEF" \
      --subtype-action-margin "$SUBTYPE_ACTION_MARGIN" \
      "${subtype_router_args[@]}" \
      --subtype-router-min-confidence "$SUBTYPE_ROUTER_MIN_CONFIDENCE" \
      --subtype-router-low-confidence-action -1 \
      --awbc-teacher-mode "$AWBC_TEACHER_MODE" \
      --awbc-teacher-event-lookahead-steps "$AWBC_TEACHER_EVENT_LOOKAHEAD_STEPS" \
      --awbc-teacher-auto-score-mode "$AWBC_TEACHER_AUTO_SCORE_MODE" \
      --awbc-teacher-subtype-calm-sensors met_station_core shielded_thermo_hygro \
      --awbc-teacher-subtype-particle-sensors met_station_core laser_disdrometer \
      --awbc-teacher-subtype-flux-sensors met_station_core fc4_flux \
      --awbc-teacher-subtype-thermal-sensors met_station_core surface_temp_ir \
      --awbc-teacher-dwell-steps "$AWBC_TEACHER_DWELL_STEPS" \
      --prior-kl-coef 0.0 \
      --greedy-lookahead-steps "$GREEDY_LOOKAHEAD_STEPS" \
      --event-start-prob 0.85 \
      "${event_aware_critic_args[@]}" \
      "${event_gated_actor_args[@]}" \
      "${context_encoder_args[@]}" \
      --context-feature-dim "$CONTEXT_FEATURE_DIM" \
      --context-hidden-dim "$CONTEXT_HIDDEN_DIM" \
      --context-fusion-mode "$CONTEXT_FUSION_MODE" \
      "${context_layer_norm_args[@]}" \
      --soc-aux-horizon 0 \
      --soc-aux-coef 0.0 \
      --learning-rate "$LEARNING_RATE" \
      --train-episode-len 512 \
      --no-use-candidate-prior \
      --candidate-prior-scale 2.0 \
      --candidate-prior-steps 512 \
      --candidate-prior-rollouts 4 \
      --static-selection-steps "$STATIC_SELECTION_STEPS" \
      --static-selection-score "$STATIC_SELECTION_SCORE" \
      --static-selection-rollouts 4 \
      --eval-steps "$EVAL_STEPS" \
      --eval-rollouts "$EVAL_ROLLOUTS" \
      --metrics-sort-score "$METRICS_SORT_SCORE" \
      --eval-start-selection subtype_balanced_transport_rich \
      --eval-event-fraction 0.75 \
      --eval-selection-stride 64 \
      --lambda-warmup-abort 1.0 \
      --lambda-switch "$LAMBDA_SWITCH" \
      --event-reward-multiplier 1.0 \
      --event-subtype-particle-reward-multiplier "$EVENT_SUBTYPE_PARTICLE_REWARD_MULTIPLIER" \
      --event-subtype-flux-reward-multiplier "$EVENT_SUBTYPE_FLUX_REWARD_MULTIPLIER" \
      --event-subtype-thermal-reward-multiplier "$EVENT_SUBTYPE_THERMAL_REWARD_MULTIPLIER" \
      --reward-loss-normalization "$REWARD_LOSS_NORMALIZATION" \
      --reward-proxy-mode "$REWARD_PROXY_MODE" \
      --lambda-duty-balance "$LAMBDA_DUTY_BALANCE" \
      --duty-balance-low "$DUTY_BALANCE_LOW" \
      --duty-balance-high "$DUTY_BALANCE_HIGH" \
      --duty-balance-grace-steps "$DUTY_BALANCE_GRACE_STEPS" \
      --duty-score-feedback "$DUTY_SCORE_FEEDBACK" \
      --duty-score-target "$DUTY_SCORE_TARGET" \
      --duty-hard-low "$DUTY_HARD_LOW" \
      --duty-hard-high "$DUTY_HARD_HIGH" \
      --duty-hard-score "$DUTY_HARD_SCORE" \
      "${duty_guard_args[@]}" \
      --no-primary-eval-duty-guard \
      --min-dwell-steps "$MIN_DWELL_STEPS" \
      "${regime_belief_args[@]}" \
      --regime-belief-lookback "$REGIME_BELIEF_LOOKBACK" \
      "${event_flag_state_args[@]}" \
      "${alert_context_args[@]}" \
      --alert-context-threshold "$ALERT_CONTEXT_THRESHOLD" \
      --alert-context-trend-lookback "$ALERT_CONTEXT_TREND_LOOKBACK" \
      --target-weights $TARGET_WEIGHTS \
      --subtype-loss-weighting \
      --subtype-particle-target-weights $SUBTYPE_PARTICLE_TARGET_WEIGHTS \
      --subtype-flux-target-weights $SUBTYPE_FLUX_TARGET_WEIGHTS \
      --subtype-thermal-target-weights $SUBTYPE_THERMAL_TARGET_WEIGHTS \
      --required-sensors met_station_core \
      --disable-coverage-groups \
      --max-active 2 \
      --agent-context-columns agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert agent_context_event_alert \
      --alert-context-columns agent_context_particle_alert agent_context_flux_alert agent_context_thermal_alert \
      --device "$DEVICE" \
      "${dry_run_args[@]}" \
      "${preflight_args[@]}" \
      2>&1 | tee "${out_dir}/run_train_eval.log"
  else
    echo "[metpair-backbone-context] seed=${seed} training artifact exists; skipping train"
  fi

  if [[ "$SKIP_ROUTER_EVAL" != "1" && ! -f "${out_dir}/eval_router_conf08/v2_custom_ppo_metrics.csv" ]]; then
    "$PY" scripts/64_v31_eval_saved_run_operational_baselines.py \
      --source-run-dir "$out_dir" \
      --out-dir "${out_dir}/eval_router_conf08" \
      --device "$DEVICE" \
      --oracle-device "$EVAL_ORACLE_DEVICE" \
      --subtype-router \
      --subtype-router-min-confidence "$EVAL_SUBTYPE_ROUTER_MIN_CONFIDENCE" \
      --skip-rollout-evaluation \
      2>&1 | tee "${out_dir}/eval_router_conf08.log"
  fi

  if [[ "$SKIP_REPLAY_GATE" != "1" && ! -f "${out_dir}/replay_gate_explicit_static_noguard/split_replay_gate_summary.json" ]]; then
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
      --lead-steps $REPLAY_LEAD_STEPS \
      --dwell-steps $REPLAY_DWELL_STEPS \
      --subtype-top-size-cap 2 \
      --static-reference-duty-guard off \
      --enforce-static-candidate-reference \
      --min-margin-abs 0.005 \
      --min-margin-rel 0.01 \
      --macro-score-column "$MACRO_SCORE_COLUMN" \
      2>&1 | tee "${out_dir}/replay_gate_explicit_static_noguard.log"
  fi

  if [[ "$SKIP_BEHAVIOR_AUDIT" != "1" && ! -f "${out_dir}/behavior_audit_v2/behavior_complexity_summary.json" ]]; then
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
