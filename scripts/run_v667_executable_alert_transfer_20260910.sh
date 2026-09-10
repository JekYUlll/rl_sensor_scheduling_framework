#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
OUT_ROOT="${V667_OUT_ROOT:-reports/v667_executable_alert_transfer_b2p15_20260910}"
mkdir -p "$OUT_ROOT"
starts=(76589 77858 79988 82985 86293 86811 88490 89496)
for seed in 7401 7402 7403 7404; do
  args=()
  for start in "${starts[@]}"; do args+=(--test-start "$start"); done
  "$PYTHON_BIN" scripts/170_audit_executable_observation_transfer.py \
    --run-dir "reports/v601_alert_coupled_assets_b2p15_20260910/seed${seed}" \
    --losses "reports/v602_alert_coupled_geometry_b2p15_20260910/seed${seed}/subset_condition_losses.csv" \
    --train-observations "reports/v603_alert_coupled_observations_b2p15_20260910/train/seed${seed}/online_observation_probe.npz" \
    --oracle "reports/v601_alert_coupled_assets_b2p15_20260910/seed${seed}/v2_tcn_oracle.pt" \
    --out-dir "$OUT_ROOT/seed${seed}" --budget 2.15 --fixed-power 0.4104 --seed "$seed" \
    "${args[@]}" > "$OUT_ROOT/seed${seed}.log" 2>&1
done
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import pandas as pd
root = Path("reports/v667_executable_alert_transfer_b2p15_20260910")
frames = [pd.read_csv(p) for p in sorted(root.glob("seed*/seed*_window_results.csv"))]
summaries = [json.loads(p.read_text()) for p in sorted(root.glob("seed*/seed*_summary.json"))]
pd.concat(frames, ignore_index=True).to_csv(root / "window_results.csv", index=False)
(root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
print(json.dumps(summaries, indent=2))
PY
