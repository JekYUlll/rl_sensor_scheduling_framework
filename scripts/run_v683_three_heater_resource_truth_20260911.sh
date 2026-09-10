#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
TRUTH_ROOT="${V683_TRUTH_ROOT:-reports/v682_joint_contrast_truth_20260911}"
OUT_ROOT="${V683_OUT_ROOT:-reports/v683_three_heater_resource_truth_20260911}"
mkdir -p "$OUT_ROOT"
for seed in 7177 7178 7179 7180; do
  "$PYTHON_BIN" scripts/190_build_v683_three_heater_resource_trace.py \
    --truth "$TRUTH_ROOT/truth_seed${seed}.csv" \
    --output "$OUT_ROOT/resource_seed${seed}.csv" \
    > "$OUT_ROOT/resource_seed${seed}.log"
done
