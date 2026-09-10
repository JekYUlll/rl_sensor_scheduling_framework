#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
INPUT_ROOT="${V682_INPUT_ROOT:-reports/v527_observable_target_truth_20260909_r2}"
OUT_ROOT="${V682_OUT_ROOT:-reports/v682_joint_contrast_truth_20260911}"
mkdir -p "$OUT_ROOT"
for seed in 7177 7178 7179 7180; do
  "$PYTHON_BIN" scripts/189_build_v682_joint_contrast_truth.py \
    --input "$INPUT_ROOT/truth_seed${seed}.csv" \
    --output "$OUT_ROOT/truth_seed${seed}.csv" \
    > "$OUT_ROOT/truth_seed${seed}.log"
done
