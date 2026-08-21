#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/bo24_local_watch_20260621.log"
STATUS_FILE="$ROOT/reports/aggregate/bo24_local_watch_20260621_status.md"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
MAX_HOURS="${MAX_HOURS:-24}"
REMOTE_ROOT="~/_code/microclimate_demo/rl_sensor_scheduling_framework"

mkdir -p "$LOG_DIR" "$ROOT/reports/aggregate"

start_epoch="$(date +%s)"
end_epoch=$((start_epoch + MAX_HOURS * 3600))

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

sync_aggregates() {
  rsync -az --include '*/' \
    --include 'metpair_backbone_context_ortholinear_balancedobjective_*_20260621*/***' \
    --include 'balancedobjective_24h_autonomy_20260621.md' \
    --exclude '*' \
    remote-gpu:~/_code/microclimate_demo/rl_sensor_scheduling_framework/reports/aggregate/ \
    "$ROOT/reports/aggregate/" >>"$LOG_FILE" 2>&1 || return 1
}

remote_snapshot() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 remote-gpu 'bash -s' <<'REMOTE'
cd ~/_code/microclimate_demo/rl_sensor_scheduling_framework || exit 1
date -Is
echo "tmux:"
tmux ls 2>/dev/null || true
echo
echo "autonomy_tail:"
tail -60 reports/aggregate/balancedobjective_24h_autonomy_20260621.md 2>/dev/null || true
echo
echo "latest_reports:"
find reports/aggregate -maxdepth 2 -type f \( -name 'BREAKTHROUGH_REPORT.md' -o -name 'oldclaim_summary.json' \) \
  -path '*balancedobjective*20260621*' -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -20
echo
echo "gpu:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
echo
echo "active_count:"
ps -fu zhangzhuyu | grep -E "25_v2_train_custom_ppo|64_v31_eval|70_v31_split|71_v31_behavior|73_v31_collect|74_v31_write|run_v31_balancedobjective" | grep -v grep | wc -l
REMOTE
}

log "watcher started; interval=${INTERVAL_SECONDS}s max_hours=${MAX_HOURS}"
while [ "$(date +%s)" -lt "$end_epoch" ]; do
  {
    echo "# BO24 Local Watch Status"
    echo
    echo "- Updated: $(date -Is)"
    echo "- Local root: $ROOT"
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

  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 remote-gpu "tmux has-session -t bo24_autonomy_20260621 2>/dev/null"; then
    log "remote bo24_autonomy_20260621 tmux session is no longer active; final sync and exit"
    sync_aggregates || true
    exit 0
  fi

  sleep "$INTERVAL_SECONDS"
done

log "watcher reached ${MAX_HOURS}h limit; final sync and exit"
sync_aggregates || true
