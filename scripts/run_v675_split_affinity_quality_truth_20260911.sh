#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
INPUT_ROOT="${V675_INPUT_ROOT:-reports/v669_shared_nowcast_truth_screen_20260910}"
OUT_ROOT="${V675_OUT_ROOT:-reports/v675_split_affinity_quality_truth_20260911}"
mkdir -p "$OUT_ROOT"
for seed in 7177 7178 7179 7180; do
  "$PYTHON_BIN" scripts/186_build_split_affinity_quality_truth.py \
    --input "$INPUT_ROOT/truth_seed${seed}.csv" \
    --output "$OUT_ROOT/truth_seed${seed}.csv"
done
