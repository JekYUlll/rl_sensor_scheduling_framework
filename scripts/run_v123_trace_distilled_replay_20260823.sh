#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/.conda/envs/darts/bin/python}"
RUN_GLOB="${RUN_GLOB:-reports/v120_full_intensity_context_gate_dev_seed*_b1p75_20260822}"
OUT_ROOT="${OUT_ROOT:-reports/aggregate/v123_trace_distilled_replay_20260823}"

cd "$ROOT"
"$PYTHON" scripts/81_v31_framework_baseline_supplements.py \
  --run-glob "$RUN_GLOB" \
  --out-root "$OUT_ROOT" \
  --oracle-device cuda \
  --policies trace_distilled
echo "V123_EXIT=0"
