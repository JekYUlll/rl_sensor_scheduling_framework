#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/behaviorbd_pilot_local_watch_20260621.log"
STATUS_FILE="$ROOT/reports/aggregate/behaviorbd_pilot_local_watch_20260621_status.md"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
MAX_HOURS="${MAX_HOURS:-24}"
SESSION="${SESSION:-behaviorbd_pilot_parallel_83_92_20260621}"

mkdir -p "$LOG_DIR" "$ROOT/reports/aggregate"

start_epoch="$(date +%s)"
end_epoch=$((start_epoch + MAX_HOURS * 3600))

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

sync_aggregates() {
  rsync -az --include '*/' \
    --include '*behaviorbd*20260621*/***' \
    --include 'behaviorbd_*_20260621.md' \
    --exclude '*' \
    remote-gpu:~/_code/microclimate_demo/rl_sensor_scheduling_framework/reports/aggregate/ \
    "$ROOT/reports/aggregate/" >>"$LOG_FILE" 2>&1 || return 1
}

remote_snapshot() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 remote-gpu 'bash -s' <<'REMOTE'
cd ~/_code/microclimate_demo/rl_sensor_scheduling_framework || exit 1
date -Is
echo "tmux:"
tmux ls 2>/dev/null | grep -E 'behaviorbd|behaviorreg|autoteacher|router|bo24' || true
echo
echo "seed_status: oracle ppo base_eval router_eval replay behavior"
for seed in 83 84 86 87 91 92; do
  d=reports/v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbd_seed${seed}_h075ctxolbd_20260621
  printf 'seed%s ' "$seed"
  for name in v2_tcn_oracle.pt custom_ppo.pt v2_custom_ppo_metrics.csv eval_router_conf08/v2_custom_ppo_metrics.csv replay_gate_explicit_static_noguard/split_replay_gate_summary.json behavior_audit_v2/behavior_complexity_summary.json; do
    test -f "$d/$name" && printf 1 || printf 0
  done
  printf '\n'
done
echo
echo "latest_logs:"
for seed in 83 84 86 87 91 92; do
  log=logs/behaviorbd_pilot_seed${seed}_20260621.log
  printf '%s | ' "$log"
  test -f "$log" && tail -n 2 "$log" | tr '\n' ' ' || printf missing
  printf '\n'
done
echo
echo "latest_reports:"
find reports/aggregate -maxdepth 2 -type f \( -name 'oldclaim_summary.json' -o -name 'metpair_claim_summary.json' -o -name '*.md' \) \
  -path '*behaviorbd*20260621*' -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -20
echo
echo "gpu:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
echo
echo "active_count:"
ps -fu zhangzhuyu | grep -E "behaviorbd|25_v2_train_custom_ppo|64_v31_eval|70_v31_split|71_v31_behavior|73_v31_collect|74_v31_write" | grep -v grep | wc -l
REMOTE
}

log "watcher started; session=${SESSION} interval=${INTERVAL_SECONDS}s max_hours=${MAX_HOURS}"
while [ "$(date +%s)" -lt "$end_epoch" ]; do
  {
    echo "# Behavior-Diversity PPO Pilot Local Watch Status"
    echo
    echo "- Updated: $(date -Is)"
    echo "- Local root: $ROOT"
    echo "- Remote session: $SESSION"
    echo
    echo '```text'
    remote_snapshot
    echo '```'
  } >"$STATUS_FILE.tmp" 2>>"$LOG_FILE" || {
    log "remote snapshot failed"
    rm -f "$STATUS_FILE.tmp"
  }
  if [ -f "$STATUS_FILE.tmp" ]; then
    mv "$STATUS_FILE.tmp" "$STATUS_FILE"
    log "remote snapshot updated"
  fi

  if sync_aggregates; then
    log "aggregate sync complete"
  else
    log "aggregate sync failed"
  fi

  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 remote-gpu "tmux has-session -t ${SESSION} 2>/dev/null"; then
    log "remote ${SESSION} tmux session is no longer active; final sync and exit"
    sync_aggregates || true
    exit 0
  fi

  sleep "$INTERVAL_SECONDS"
done

log "watcher reached ${MAX_HOURS}h limit; final sync and exit"
sync_aggregates || true
