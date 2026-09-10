#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
GEOM_TRAIN="${V664_TRAIN_ROOT:-reports/v661_observable_specialist_mode_geometry_mode_labels_20260910}"
GEOM_TEST="${V664_TEST_ROOT:-reports/v662_observable_specialist_mode_geometry_testgrid_20260910}"
OBS_ROOT="${V664_OBS_ROOT:-reports/v663_observable_specialist_mode_observations_20260910}"
TRACE_ROOT="${V664_TRACE_ROOT:-reports/v658_observable_specialist_mode_truth_20260910/resource}"
OUT_ROOT="${V664_OUT_ROOT:-reports/v664_observable_specialist_mode_transfer_20260910}"
mkdir -p "$OUT_ROOT"
for seed in 7231 7232 7233 7234; do
  "$PYTHON_BIN" - "$seed" "$GEOM_TRAIN" "$GEOM_TEST" "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd
seed, train_root, test_root, out_root = sys.argv[1:]
train_paths = sorted(Path(train_root, f"seed{seed}").glob("train/subset_condition_losses_seed*.csv"))
test_paths = sorted(Path(test_root, f"seed{seed}").glob("test_*/subset_condition_losses_seed*.csv"))
if not train_paths or not test_paths:
    raise SystemExit(f"missing geometry loss files for seed {seed}: {train_paths} {test_paths}")
out = Path(out_root)
out.mkdir(parents=True, exist_ok=True)
pd.concat([pd.read_csv(p) for p in train_paths], ignore_index=True).to_csv(out / f"train_losses_seed{seed}.csv", index=False)
pd.concat([pd.read_csv(p) for p in test_paths], ignore_index=True).to_csv(out / f"test_losses_seed{seed}.csv", index=False)
PY
  "$PYTHON_BIN" scripts/169_audit_full_observation_transfer.py \
    --train-losses "$OUT_ROOT/train_losses_seed${seed}.csv" \
    --test-losses "$OUT_ROOT/test_losses_seed${seed}.csv" \
    --train-observations "$OBS_ROOT/train/seed${seed}/online_observation_probe.npz" \
    --test-observations "$OBS_ROOT/test/seed${seed}/online_observation_probe.npz" \
    --train-trace "$TRACE_ROOT/resource_seed${seed}.csv" \
    --test-trace "$TRACE_ROOT/resource_seed${seed}.csv" \
    --effective-budget 5.0 --fixed-power 0.4104 --seed "$seed" \
    --output "$OUT_ROOT/seed${seed}/transfer" > "$OUT_ROOT/seed${seed}.log" 2>&1
done
