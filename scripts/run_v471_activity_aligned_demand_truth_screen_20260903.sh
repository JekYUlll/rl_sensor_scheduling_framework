#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="reports/analysis/v471_activity_aligned_demand_truth_screen_20260903"
test ! -e "$OUT"
TRUTHS=()
for seed in 7091 7092; do
  run="reports/v470_activity_aligned_demand_truth_seed${seed}_b1p75_20260822"
  test -s "$run/truth_v31_split.csv"
  test ! -e "$run/custom_ppo.pt"
  TRUTHS+=(--truth-csv "$run/truth_v31_split.csv")
done
"${PYTHON_BIN:-$HOME/.conda/envs/darts/bin/python}" \
  scripts/120_v32_audit_forecast_value_truth.py \
  "${TRUTHS[@]}" --lead-steps 8 --activity-aligned-transport-demand \
  --out "$OUT/truth_gate.json"
