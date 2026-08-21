#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATE_TAG="${DATE_TAG:-20260622}"
SEED_LABEL="${SEED_LABEL:-141_142_143_144_145_146}"
VARIANT="${VARIANT:-weaker_latent10_actionaux_ce05_margin005}"

STATUS_OUT="reports/aggregate/actionaux_pilot_${SEED_LABEL}_status_${DATE_TAG}.md"
mkdir -p "$(dirname "$STATUS_OUT")"

{
  printf '# Action-Aux Pilot Status\n\n'
  printf 'Generated: %s\n\n' "$(date -Is)"

  printf '## Tmux\n\n'
  tmux ls 2>/dev/null | grep 'actionaux_20260622' || true
  printf '\n'

  printf '## Top-Level Log Tail\n\n'
  if [[ -f "logs/actionaux_tmux_${DATE_TAG}.log" ]]; then
    tail -80 "logs/actionaux_tmux_${DATE_TAG}.log"
  else
    printf 'No top-level log yet.\n'
  fi
  printf '\n'

  printf '## Seed Progress\n\n'
  for f in logs/actionaux_${VARIANT}_seed*_${DATE_TAG}.log; do
    [[ -f "$f" ]] || continue
    printf '%s :: ' "$(basename "$f")"
    grep -E 'custom_ppo_update|actionaux_seed_done|Traceback|Error|Exception' "$f" | tail -1 || true
  done
  printf '\n'

  printf '## Decision Audits\n\n'
  find reports/aggregate -maxdepth 1 -type f -name "actionaux_${VARIANT}_${SEED_LABEL}_decision_audit_${DATE_TAG}.json" -print | sort
  printf '\n'

  printf '## Summary\n\n'
  summary="reports/aggregate/actionaux_pilot_${SEED_LABEL}_${DATE_TAG}/robustness_summary.md"
  if [[ -f "$summary" ]]; then
    cat "$summary"
  else
    printf 'No final action-aux summary yet.\n'
  fi
} | tee "$STATUS_OUT"

echo "wrote $STATUS_OUT"
