#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
RUN_ROOT="${V663_RUN_ROOT:-reports/v659_observable_specialist_mode_assets_20260910}"
OUT_ROOT="${V663_OUT_ROOT:-reports/v663_observable_specialist_mode_observations_20260910}"
mkdir -p "$OUT_ROOT"
train_args=(--start-index 77452 --start-index 78907 --start-index 80344)
test_args=(--start-index 82600 --start-index 83900 --start-index 85200 --start-index 86500 --start-index 87800)
for seed in 7231 7232 7233 7234; do
  "$PYTHON_BIN" scripts/168_dump_online_observation_probe.py \
    --run-dir "$RUN_ROOT/seed${seed}" --out-dir "$OUT_ROOT/train/seed${seed}" \
    --steps 256 --dynamic-resource-budget 5.0 --dynamic-resource-fixed-power 0.4104 \
    "${train_args[@]}" > "$OUT_ROOT/train_seed${seed}.log" 2>&1
  "$PYTHON_BIN" scripts/168_dump_online_observation_probe.py \
    --run-dir "$RUN_ROOT/seed${seed}" --out-dir "$OUT_ROOT/test/seed${seed}" \
    --steps 256 --dynamic-resource-budget 5.0 --dynamic-resource-fixed-power 0.4104 \
    "${test_args[@]}" > "$OUT_ROOT/test_seed${seed}.log" 2>&1
done
