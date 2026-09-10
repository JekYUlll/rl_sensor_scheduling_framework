#!/usr/bin/env bash
set -euo pipefail

# Truth-only screen.  It intentionally stops before asset fitting, geometry,
# and PPO so that the new target coupling is accepted only by a predeclared
# physics/forecast audit.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
INPUT_ROOT="${V645_INPUT_ROOT:-reports/v614_empirical_cold_availability_truth_20260910/truth}"
OUT_ROOT="${V645_OUT_ROOT:-reports/v645_empirical_cold_wind_truth_20260910}"
mkdir -p "$OUT_ROOT/truth"

for seed in 7401 7402 7403 7404; do
  "$PYTHON_BIN" scripts/183_build_empirical_cold_wind_innovation_truth.py \
    --input "$INPUT_ROOT/truth_seed${seed}_empirical_cold.csv" \
    --output "$OUT_ROOT/truth/truth_seed${seed}_cold_wind.csv" \
    --seed "$seed" \
    --lead-steps 6
done

"$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "truth").glob("*.json")):
    rows.append(json.loads(path.read_text(encoding="utf-8")))
(root / "truth_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print(json.dumps(rows, indent=2))
PY
