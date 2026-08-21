#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/behaviorbrg_pilot_local_watch_20260621.log"
STATUS_FILE="$ROOT/reports/aggregate/behaviorbrg_pilot_local_watch_20260621_status.md"
REMOTE_DIR="~/_code/microclimate_demo/rl_sensor_scheduling_framework"
SESSION="${SESSION:-behaviorbrg_pilot_parallel_83_92_20260621}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
mkdir -p "$LOG_DIR" "$ROOT/reports/aggregate"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

remote_snapshot() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 remote-gpu "cd ${REMOTE_DIR} && bash -s" <<'REMOTE'
printf 'REMOTE_DATE '; date '+%Y-%m-%dT%H:%M:%S%z'
printf '\nTMUX\n'
tmux ls 2>/dev/null | grep -E 'behaviorbrg|behaviorbd|behaviorreg|autoteacher|router|bo24' || true
printf '\nSTATUS bits: oracle ppo base_eval router_eval replay behavior\n'
for seed in 83 84 86 87 91 92; do
  d=reports/v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg_seed${seed}_h075ctxolbrg_20260621
  printf 'seed%s ' "$seed"
  for name in v2_tcn_oracle.pt custom_ppo.pt v2_custom_ppo_metrics.csv eval_router_conf08/v2_custom_ppo_metrics.csv replay_gate_explicit_static_noguard/split_replay_gate_summary.json behavior_audit_v2/behavior_complexity_summary.json; do
    test -f "$d/$name" && printf 1 || printf 0
  done
  printf ' '
  if test -f "logs/behaviorbrg_pilot_seed${seed}_20260621.log"; then
    grep -Eo 'timesteps=[0-9]+' "logs/behaviorbrg_pilot_seed${seed}_20260621.log" | tail -1 | tr -d '\n' || true
  fi
  printf '\n'
done
printf '\nAGG FILES\n'
find reports/aggregate -maxdepth 2 -type f \( -name 'oldclaim_summary.json' -o -name 'metpair_claim_summary.json' -o -name '*.md' \) \
  -path '*behaviorbrg*20260621*' -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -20
printf '\nLOG TAILS\n'
for seed in 83 84 86 87 91 92; do
  log=logs/behaviorbrg_pilot_seed${seed}_20260621.log
  printf '%s | ' "$log"
  test -f "$log" && tail -n 3 "$log" | tr '\n' ' ' || printf missing
  printf '\n'
done
printf '\nPYTHON PROCESSES\n'
ps -fu zhangzhuyu | grep -E "behaviorbrg|25_v2_train_custom_ppo|64_v31_eval|70_v31_split|71_v31_behavior|73_v31_collect|74_v31_write" | grep -v grep | wc -l
printf '\nGPU\n'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
REMOTE
}

sync_aggregates() {
  rsync -az \
    --include '*behaviorbrg*20260621*/***' \
    --include 'behaviorbrg_*_20260621.md' \
    --exclude '*' \
    remote-gpu:~/_code/microclimate_demo/rl_sensor_scheduling_framework/reports/aggregate/ \
    "$ROOT/reports/aggregate/" >>"$LOG_FILE" 2>&1 || return 1
}

write_status() {
  {
    printf '# BRG-1 Pilot Watch Status\n\n'
    printf '- Local time: `%s`\n' "$(date -Is)"
    printf '- Remote alias: `remote-gpu`\n'
    printf '- Remote session: `%s`\n\n' "$SESSION"
    remote_snapshot
  } >"$STATUS_FILE.tmp"
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

while true; do
  log "watch tick"
  if sync_aggregates; then
    log "aggregate sync complete"
  else
    log "aggregate sync failed"
  fi
  write_status || log "status write failed"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=20 remote-gpu "tmux has-session -t ${SESSION}" >/dev/null 2>&1; then
    log "remote session ${SESSION} not active; final sync and exit"
    sync_aggregates || true
    write_status || true
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
