#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
SOURCE_ROOT="${V640_SOURCE_ROOT:-reports/v525_stage_b_truth_resource_trace_20260909}"
OUT_ROOT="${V640_OUT_ROOT:-reports/v640_radiometer_signal_truth_20260910}"
cd "$ROOT"
mkdir -p "$OUT_ROOT"
for seed in 7177 7178 7179 7180; do
  "$PYTHON_BIN" scripts/181_build_observable_operating_truth.py \
    --input "$SOURCE_ROOT/resource_trace_seed${seed}.csv" \
    --output "$OUT_ROOT/truth_seed${seed}.csv" \
    --use-nowcast-resource --heater-quality-coupling --radiometer-signal-coupling \
    > "$OUT_ROOT/truth_seed${seed}.summary.json"
done
"$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
for truth_path in sorted(root.glob("truth_seed*.csv")):
    d = pd.read_csv(truth_path)
    cols = [c for c in d.columns if c == "time_idx" or c.startswith("resource_")]
    d[cols].to_csv(root / truth_path.name.replace("truth_", "resource_"), index=False)
PY
