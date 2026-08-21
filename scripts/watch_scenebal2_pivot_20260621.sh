#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="~/_code/microclimate_demo/rl_sensor_scheduling_framework"
SESSION="${SESSION:-scenebal2_pivot_122_117_20260621}"
DATE_TAG="${DATE_TAG:-20260621}"
SEEDS="${SEEDS:-122 117}"
SEED_LABEL="${SEED_LABEL:-122_117}"
RUN_PREFIX="${RUN_PREFIX:-v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2}"
BUDGET_LABEL="${BUDGET_LABEL:-h075ctxolscbal2}"
EVAL_DIR="${EVAL_DIR:-eval_router_conf05_scenebal2_20260621}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-behavior_audit_router_conf05_scenebal2_20260621}"
AGG_LABEL="${AGG_LABEL:-scenebal2_pivot_conf05_${SEED_LABEL}}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/scenebal2_pivot_${SEED_LABEL}_local_watch_${DATE_TAG}.log"
STATUS_FILE="$ROOT/reports/aggregate/scenebal2_pivot_${SEED_LABEL}_local_watch_${DATE_TAG}_status.md"

mkdir -p "$LOG_DIR" "$ROOT/reports/aggregate"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

remote_snapshot() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 remote-gpu "cd ${REMOTE_DIR} && SESSION_NAME=\"${SESSION}\" SEEDS_REMOTE=\"${SEEDS}\" RUN_PREFIX_REMOTE=\"${RUN_PREFIX}\" BUDGET_LABEL_REMOTE=\"${BUDGET_LABEL}\" DATE_TAG_REMOTE=\"${DATE_TAG}\" EVAL_DIR_REMOTE=\"${EVAL_DIR}\" BEHAVIOR_DIR_REMOTE=\"${BEHAVIOR_DIR}\" AGG_LABEL_REMOTE=\"${AGG_LABEL}\" bash -s" <<'REMOTE'
printf 'REMOTE_DATE '; date '+%Y-%m-%dT%H:%M:%S%z'
printf '\nTMUX_HAS_SESSION '
if tmux has-session -t "$SESSION_NAME" >/dev/null 2>&1; then
  printf 'yes\n'
else
  printf 'no\n'
fi
printf '\nGPU\n'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader || true
printf '\nARTIFACT STATUS bits: oracle ppo base_eval router_eval replay behavior\n'
for seed in $SEEDS_REMOTE; do
  d=reports/${RUN_PREFIX_REMOTE}_seed${seed}_${BUDGET_LABEL_REMOTE}_${DATE_TAG_REMOTE}
  printf 'seed%s ' "$seed"
  for name in v2_tcn_oracle.pt custom_ppo.pt v2_custom_ppo_metrics.csv ${EVAL_DIR_REMOTE}/v2_custom_ppo_metrics.csv replay_gate_explicit_static_noguard/split_replay_gate_summary.json ${BEHAVIOR_DIR_REMOTE}/behavior_complexity_summary.json; do
    test -f "$d/$name" && printf 1 || printf 0
  done
  printf '\n'
done
printf '\nTRAINING PROGRESS latest_timestep log_bytes latest_key_line\n'
for seed in $SEEDS_REMOTE; do
  d=reports/${RUN_PREFIX_REMOTE}_seed${seed}_${BUDGET_LABEL_REMOTE}_${DATE_TAG_REMOTE}
  log="$d/run_train_eval.log"
  printf 'seed%s ' "$seed"
  if [ -f "$log" ]; then
    ts="$(grep -Eo 'timesteps=[0-9]+' "$log" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
    if [ -z "$ts" ] && grep -q 'custom_ppo_bc_pretrain' "$log" 2>/dev/null; then
      ts="bc_pretrain_done"
    fi
    bytes="$(wc -c <"$log" 2>/dev/null || printf 0)"
    line="$(grep -E 'custom_ppo_update|custom_ppo_bc_pretrain|custom_ppo_eval|Traceback|ERROR|RuntimeError|CUDA out of memory' "$log" 2>/dev/null | tail -1 | tr '|' ' ' | cut -c1-220 || true)"
    printf '%s %s %s\n' "${ts:-not_started}" "${bytes:-0}" "${line:-no_key_line}"
  else
    printf 'no_log 0 no_log\n'
  fi
done
printf '\nRECENT SCENEBAL-2 AGGREGATES\n'
find reports/aggregate -maxdepth 2 -type f \
  -path "*${AGG_LABEL_REMOTE}*${DATE_TAG_REMOTE}*" \
  -printf '%TY-%Tm-%Td %TH:%M %p\n' 2>/dev/null | sort | tail -40
printf '\nERROR SCAN\n'
grep -RInE 'Traceback|ERROR|Exception|RuntimeError|CUDA out of memory|nan' \
  logs/scenebal2_pivot_*_${DATE_TAG_REMOTE}.log \
  reports/${RUN_PREFIX_REMOTE}_seed*_${BUDGET_LABEL_REMOTE}_${DATE_TAG_REMOTE}/run_train_eval.log \
  2>/dev/null | tail -40 || true
REMOTE
}

write_status() {
  {
    printf '# SCENEBAL-2 Pivot Watch Status\n\n'
    printf -- '- Local time: `%s`\n' "$(date -Is)"
    printf -- '- Remote alias: `remote-gpu`\n'
    printf -- '- Remote session: `%s`\n' "$SESSION"
    printf -- '- Seeds: `%s`\n' "$SEEDS"
    printf -- '- Aggregate label: `%s`\n\n' "$AGG_LABEL"
    remote_snapshot
  } >"$STATUS_FILE.tmp"
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

sync_outputs() {
  rsync -az \
    --include "${AGG_LABEL}_*_${DATE_TAG}/***" \
    --include "${AGG_LABEL}_*_${DATE_TAG}.json" \
    --include "${AGG_LABEL}_*_${DATE_TAG}.md" \
    --include "${AGG_LABEL}_*_${DATE_TAG}.csv" \
    --exclude '*' \
    remote-gpu:~/_code/microclimate_demo/rl_sensor_scheduling_framework/reports/aggregate/ \
    "$ROOT/reports/aggregate/" >>"$LOG_FILE" 2>&1 || return 1

  for seed in $SEEDS; do
    run_dir="${RUN_PREFIX}_seed${seed}_${BUDGET_LABEL}_${DATE_TAG}"
    mkdir -p "$ROOT/reports/${run_dir}"
    rsync -az \
      --include "${run_dir}/" \
      --include "${run_dir}/v2_custom_ppo_metrics.csv" \
      --include "${run_dir}/split_static_candidate_event_table.csv" \
      --include "${run_dir}/validation_static_candidates.csv" \
      --include "${run_dir}/run_train_eval.log" \
      --include "${run_dir}/${EVAL_DIR}/" \
      --include "${run_dir}/${EVAL_DIR}/v2_custom_ppo_metrics.csv" \
      --include "${run_dir}/replay_gate_explicit_static_noguard/" \
      --include "${run_dir}/replay_gate_explicit_static_noguard/split_replay_gate_summary.json" \
      --include "${run_dir}/${BEHAVIOR_DIR}/" \
      --include "${run_dir}/${BEHAVIOR_DIR}/behavior_complexity_summary.json" \
      --exclude '*' \
      remote-gpu:~/_code/microclimate_demo/rl_sensor_scheduling_framework/reports/ \
      "$ROOT/reports/" >>"$LOG_FILE" 2>&1 || return 1
  done

  rsync -az \
    --include "scenebal2_pivot_*_${DATE_TAG}.log" \
    --include "scenebal2_pivot_${SEED_LABEL}_${DATE_TAG}.master.log" \
    --exclude '*' \
    remote-gpu:~/_code/microclimate_demo/rl_sensor_scheduling_framework/logs/ \
    "$ROOT/logs/" >>"$LOG_FILE" 2>&1 || true
}

aggregate_ready() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 remote-gpu "cd ${REMOTE_DIR} && AGG_LABEL_REMOTE=\"${AGG_LABEL}\" DATE_TAG_REMOTE=\"${DATE_TAG}\" bash -s" <<'REMOTE'
set -euo pipefail
test -f "reports/aggregate/${AGG_LABEL_REMOTE}_decision_audit_${DATE_TAG_REMOTE}.json"
REMOTE
}

while true; do
  log "scenebal2 watch tick"
  sync_outputs || log "sync outputs failed or incomplete"
  write_status || log "status write failed"
  set +e
  ssh -o BatchMode=yes -o ConnectTimeout=20 remote-gpu "tmux has-session -t ${SESSION}" >/dev/null 2>&1
  session_rc=$?
  set -e
  if [[ "$session_rc" -eq 0 ]]; then
    log "remote session still active; waiting"
    sleep "$INTERVAL_SECONDS"
    continue
  fi
  if [[ "$session_rc" -eq 255 ]]; then
    log "remote session check failed with ssh rc=255; treating as transient and retrying"
    sleep "$INTERVAL_SECONDS"
    continue
  fi
  if aggregate_ready >>"$LOG_FILE" 2>&1; then
    log "remote session ended and decision aggregate exists; final sync"
    sync_outputs || log "final sync failed"
    write_status || true
    exit 0
  fi
  log "remote session ended but decision aggregate is not present; retrying"
  sleep "$INTERVAL_SECONDS"
done
