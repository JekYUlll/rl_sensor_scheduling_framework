#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATE_TAG="${DATE_TAG:-20260622}"
SEEDS_TEXT="${SEEDS_TEXT:-141 142 143 144 145 146}"
SEED_LABEL="${SEED_LABEL:-$(printf '%s' "${SEEDS_TEXT}" | tr ' ' '_')}"
STATUS_PATH="reports/aggregate/robustness_pilot_${SEED_LABEL}_status_${DATE_TAG}.md"

mkdir -p "$(dirname "${STATUS_PATH}")"

{
  echo "# Robustness Pilot Status"
  echo
  echo "Generated: $(date -Is)"
  echo
  echo "## Tmux"
  echo
  tmux ls 2>/dev/null | grep -E "robustness|mech_ablation" || true
  echo
  echo "## Top-Level Log Tail"
  echo
  if [[ -f "logs/robustness_tmux_${DATE_TAG}.log" ]]; then
    tail -80 "logs/robustness_tmux_${DATE_TAG}.log"
  else
    echo "No robustness tmux log yet."
  fi
  echo
  echo "## Seed Progress"
  echo
  for f in logs/robustness_*_seed*_"${DATE_TAG}".log; do
    [[ -f "${f}" ]] || continue
    printf '%s :: ' "$(basename "${f}")"
    grep "custom_ppo_update" "${f}" | tail -1 || true
  done
  echo
  echo "## Decision Audits"
  echo
  ls -1 reports/aggregate/robustness_*_"${SEED_LABEL}"_decision_audit_"${DATE_TAG}".json 2>/dev/null || true
  echo
  echo "## Summary"
  echo
  if [[ -f "reports/aggregate/robustness_pilot_${SEED_LABEL}_${DATE_TAG}/robustness_summary.md" ]]; then
    cat "reports/aggregate/robustness_pilot_${SEED_LABEL}_${DATE_TAG}/robustness_summary.md"
  else
    echo "No final robustness summary yet."
  fi
} | tee "${STATUS_PATH}"

echo "wrote ${STATUS_PATH}"
