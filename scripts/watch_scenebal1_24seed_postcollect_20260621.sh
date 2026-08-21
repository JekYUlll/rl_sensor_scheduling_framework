#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="~/_code/microclimate_demo/rl_sensor_scheduling_framework"
SESSION="${SESSION:-scenebal1_waitfree_111_116_20260621}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
DATE_TAG="${DATE_TAG:-20260621}"
SEEDS="${SEEDS:-93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116}"
SEED_LABEL="${SEED_LABEL:-93_116}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/scenebal1_24seed_postcollect_${SEED_LABEL}_${DATE_TAG}.log"
STATUS_FILE="$ROOT/reports/aggregate/scenebal1_24seed_${SEED_LABEL}_postcollect_status_${DATE_TAG}.md"

mkdir -p "$LOG_DIR" "$ROOT/reports/aggregate"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

write_status() {
  {
    printf '# SCENEBAL-1 24-Seed Post-Collect Watch\n\n'
    printf -- '- Local time: `%s`\n' "$(date -Is)"
    printf -- '- Remote session: `%s`\n' "$SESSION"
    printf -- '- Seed label: `%s`\n' "$SEED_LABEL"
    printf -- '- Seeds: `%s`\n\n' "$SEEDS"
    if ! ssh -o BatchMode=yes -o ConnectTimeout=20 remote-gpu "cd ${REMOTE_DIR} && SESSION_NAME=\"${SESSION}\" SEEDS_REMOTE=\"${SEEDS}\" bash -s" 2>/dev/null <<'REMOTE'
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
  d=reports/v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal1_seed${seed}_h075ctxolscbal1_20260621
  printf 'seed%s ' "$seed"
  for name in v2_tcn_oracle.pt custom_ppo.pt v2_custom_ppo_metrics.csv eval_router_conf08/v2_custom_ppo_metrics.csv replay_gate_explicit_static_noguard/split_replay_gate_summary.json behavior_audit_v2/behavior_complexity_summary.json; do
    test -f "$d/$name" && printf 1 || printf 0
  done
  printf '\n'
done
printf '\nTRAINING PROGRESS latest_timestep log_bytes latest_key_line\n'
for seed in $SEEDS_REMOTE; do
  d=reports/v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal1_seed${seed}_h075ctxolscbal1_20260621
  log="$d/run_train_eval.log"
  printf 'seed%s ' "$seed"
  if [ -f "$log" ]; then
    ts="$(grep -Eo 'timesteps=[0-9]+' "$log" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
    if [ -z "$ts" ] && grep -q 'custom_ppo_bc_pretrain' "$log" 2>/dev/null; then
      ts="bc_pretrain_done"
    fi
    bytes="$(wc -c <"$log" 2>/dev/null || printf 0)"
    line="$(grep -E 'custom_ppo_update|custom_ppo_bc_pretrain|custom_ppo_eval|Traceback|ERROR|RuntimeError|CUDA out of memory' "$log" 2>/dev/null | tail -1 | tr '|' ' ' | cut -c1-180 || true)"
    printf '%s %s %s\n' "${ts:-not_started}" "${bytes:-0}" "${line:-no_key_line}"
  else
    printf 'no_log 0 no_log\n'
  fi
done
printf '\nRECENT 24-SEED AGGREGATES\n'
find reports/aggregate -maxdepth 2 -type f \
  -path '*scenebal1_24seed_93_116*20260621*' \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -30
REMOTE
    then
      printf '\nREMOTE_STATUS_ERROR ssh status query failed; will retry on next watcher tick\n'
    fi
  } >"$STATUS_FILE.tmp"
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

sync_24seed() {
  rsync -az \
    --include "scenebal1_24seed_${SEED_LABEL}_*_${DATE_TAG}/***" \
    --include "scenebal1_24seed_${SEED_LABEL}_decision_audit_${DATE_TAG}.json" \
    --include "scenebal1_24seed_${SEED_LABEL}_decision_audit_${DATE_TAG}.md" \
    --include "scenebal1_24seed_${SEED_LABEL}_next_action_protocol_${DATE_TAG}.md" \
    --include "scenebal1_24seed_${SEED_LABEL}_postcollect_status_${DATE_TAG}.md" \
    --exclude '*' \
    remote-gpu:~/_code/microclimate_demo/rl_sensor_scheduling_framework/reports/aggregate/ \
    "$ROOT/reports/aggregate/" >>"$LOG_FILE" 2>&1 || return 1
}

run_remote_collect() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 remote-gpu "cd ${REMOTE_DIR} && SEEDS_REMOTE=\"${SEEDS}\" SEED_LABEL_REMOTE=\"${SEED_LABEL}\" DATE_TAG_REMOTE=\"${DATE_TAG}\" bash -s" <<'REMOTE'
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate darts

RUNS=()
for seed in $SEEDS_REMOTE; do
  RUNS+=("reports/v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal1_seed${seed}_h075ctxolscbal1_${DATE_TAG_REMOTE}")
done

MACRO_DIR="reports/aggregate/scenebal1_24seed_${SEED_LABEL_REMOTE}_macro_${DATE_TAG_REMOTE}"
RAW_MACRO_DIR="reports/aggregate/scenebal1_24seed_${SEED_LABEL_REMOTE}_raw_macro_${DATE_TAG_REMOTE}"
OLD_DIR="reports/aggregate/scenebal1_24seed_${SEED_LABEL_REMOTE}_oldclaim_replaynorm_${DATE_TAG_REMOTE}"

python scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --out-dir "${MACRO_DIR}" \
  2>&1 | tee "logs/scenebal1_24seed_collect_macro_${SEED_LABEL_REMOTE}_${DATE_TAG_REMOTE}.log"

python scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${RUNS[@]}" \
  --router-eval-dir . \
  --out-dir "${RAW_MACRO_DIR}" \
  2>&1 | tee "logs/scenebal1_24seed_collect_raw_macro_${SEED_LABEL_REMOTE}_${DATE_TAG_REMOTE}.log"

python scripts/73_v31_collect_oldclaim_gate.py \
  --runs "${RUNS[@]}" \
  --out-dir "${OLD_DIR}" \
  2>&1 | tee "logs/scenebal1_24seed_collect_oldclaim_${SEED_LABEL_REMOTE}_${DATE_TAG_REMOTE}.log"

python scripts/74_v31_write_balancedobjective_report.py \
  --macro-dir "${MACRO_DIR}" \
  --raw-macro-dir "${RAW_MACRO_DIR}" \
  --oldclaim-dir "${OLD_DIR}" \
  --out-file "${OLD_DIR}/SCENEBAL1_24SEED_REPORT.md" \
  --title "SCENEBAL-1 24-Seed Stress Aggregate Report" \
  --notes "Post-stress aggregate over seeds ${SEEDS_REMOTE}; corrected replay-normalized true-static macro comparison." \
  2>&1 | tee "logs/scenebal1_24seed_write_report_${SEED_LABEL_REMOTE}_${DATE_TAG_REMOTE}.log"

python scripts/75_v31_decide_scenebal1_stress_claim.py \
  --oldclaim-dir "${OLD_DIR}" \
  --macro-dir "${MACRO_DIR}" \
  --raw-macro-dir "${RAW_MACRO_DIR}" \
  --expected-seeds 24 \
  --label "SCENEBAL-1 24-Seed Stress Aggregate" \
  --out-json "reports/aggregate/scenebal1_24seed_${SEED_LABEL_REMOTE}_decision_audit_${DATE_TAG_REMOTE}.json" \
  --out-md "reports/aggregate/scenebal1_24seed_${SEED_LABEL_REMOTE}_decision_audit_${DATE_TAG_REMOTE}.md" \
  2>&1 | tee "logs/scenebal1_24seed_decision_audit_${SEED_LABEL_REMOTE}_${DATE_TAG_REMOTE}.log"

python scripts/76_v31_write_next_action_protocol.py \
  --decision-json "reports/aggregate/scenebal1_24seed_${SEED_LABEL_REMOTE}_decision_audit_${DATE_TAG_REMOTE}.json" \
  --out-md "reports/aggregate/scenebal1_24seed_${SEED_LABEL_REMOTE}_next_action_protocol_${DATE_TAG_REMOTE}.md" \
  --label "SCENEBAL-1 24-Seed Stress Next Action" \
  2>&1 | tee "logs/scenebal1_24seed_next_action_${SEED_LABEL_REMOTE}_${DATE_TAG_REMOTE}.log"
REMOTE
}

all_artifacts_ready() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 remote-gpu "cd ${REMOTE_DIR} && SEEDS_REMOTE=\"${SEEDS}\" bash -s" <<'REMOTE'
set -euo pipefail
missing=0
for seed in $SEEDS_REMOTE; do
  d=reports/v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal1_seed${seed}_h075ctxolscbal1_20260621
  for name in v2_tcn_oracle.pt custom_ppo.pt v2_custom_ppo_metrics.csv eval_router_conf08/v2_custom_ppo_metrics.csv replay_gate_explicit_static_noguard/split_replay_gate_summary.json behavior_audit_v2/behavior_complexity_summary.json; do
    if [ ! -f "$d/$name" ]; then
      printf 'missing seed%s %s\n' "$seed" "$name"
      missing=1
    fi
  done
done
exit "$missing"
REMOTE
}

while true; do
  log "postcollect watch tick"
  write_status || log "status write failed"
  if ssh -o BatchMode=yes -o ConnectTimeout=20 remote-gpu "tmux has-session -t ${SESSION}" >/dev/null 2>&1; then
    log "remote session still active; waiting"
    sleep "$INTERVAL_SECONDS"
    continue
  fi
  if ! all_artifacts_ready >>"$LOG_FILE" 2>&1; then
    log "remote session ended but required seed artifacts are incomplete; waiting"
    sleep "$INTERVAL_SECONDS"
    continue
  fi
  log "remote session ended; running 24-seed collect"
  if run_remote_collect >>"$LOG_FILE" 2>&1; then
    log "remote 24-seed collect complete"
    sync_24seed || log "24-seed sync failed"
    write_status || true
    exit 0
  fi
  log "remote 24-seed collect failed; retrying after interval"
  sleep "$INTERVAL_SECONDS"
done
