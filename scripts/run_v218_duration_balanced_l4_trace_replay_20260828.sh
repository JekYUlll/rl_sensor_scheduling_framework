#!/usr/bin/env bash
set -euo pipefail

# Offline diagnostic only. The policy is fitted to policy-training l4 receding
# costs, then evaluated once on the frozen final partitions under normal env
# state transitions and feasibility rules.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/.conda/envs/darts/bin/python}"
RUN_PREFIX="${RUN_PREFIX:-v213_duration_balanced_scene_dev}"
RUN_SUFFIX="${RUN_SUFFIX:-b1p85_20260822}"
OUT_ROOT="${OUT_ROOT:-reports/aggregate/v218_duration_balanced_l4_trace_replay_20260828}"

"$PYTHON" scripts/81_v31_framework_baseline_supplements.py \
  --run-glob "reports/${RUN_PREFIX}_seed*_${RUN_SUFFIX}" \
  --out-root "$OUT_ROOT" \
  --router-eval-dir __none__ \
  --replay-dir __none__ \
  --oracle-device cuda \
  --policies trace_distilled \
  --trace-training-subdir receding_oracle_l4_rl_train_trace
