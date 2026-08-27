#!/usr/bin/env bash
set -euo pipefail

# Offline diagnostic only. These labels use future information and are never
# exposed to a deployed policy or used as PD-PPO training targets.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/.conda/envs/darts/bin/python}"
RUN_PREFIX="${RUN_PREFIX:-v213_duration_balanced_scene_dev}"
RUN_SUFFIX="${RUN_SUFFIX:-b1p85_20260822}"
OUT_ROOT="${OUT_ROOT:-reports/aggregate/v216_duration_balanced_lookahead_20260828}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-2301 2302 2303 2304 2305}"
read -r -a LOOKAHEADS <<< "${LOOKAHEADS_OVERRIDE:-0 2 4 8}"

mkdir -p "$OUT_ROOT" logs
for lookahead in "${LOOKAHEADS[@]}"; do
  pids=()
  for idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[$idx]}"
    run_dir="reports/${RUN_PREFIX}_seed${seed}_${RUN_SUFFIX}"
    (
      export CUDA_VISIBLE_DEVICES="$idx"
      "$PYTHON" scripts/99_v32_receding_upper.py \
        --run-dir "$run_dir" \
        --partition final_test \
        --output-subdir "receding_oracle_l${lookahead}_lookahead_sweep" \
        --receding-oracle-lookahead-steps "$lookahead" \
        --device cuda
    ) >"logs/v216_lookahead_l${lookahead}_seed${seed}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

"$PYTHON" - "$OUT_ROOT" "${SEEDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
seeds = [int(seed) for seed in sys.argv[2:]]
rows = []
for lookahead in (0, 2, 4, 8):
    for seed in seeds:
        path = Path(
            f"reports/v213_duration_balanced_scene_dev_seed{seed}_b1p85_20260822/"
            f"receding_oracle_l{lookahead}_lookahead_sweep/oracle_lift_summary.json"
        )
        payload = json.loads(path.read_text())
        static = float(payload["validation_selected_static"]["oracle_loss_mean"])
        receding = float(payload["receding_oracle"]["oracle_loss_mean"])
        rows.append({
            "lookahead_steps": lookahead,
            "seed": seed,
            "static_loss": static,
            "receding_loss": receding,
            "static_minus_receding": static - receding,
        })
df = pd.DataFrame(rows).sort_values(["lookahead_steps", "seed"])
summary = (
    df.groupby("lookahead_steps", as_index=False)
    .agg(
        mean_static_minus_receding=("static_minus_receding", "mean"),
        wins=("static_minus_receding", lambda x: int((x > 0).sum())),
        seeds=("seed", "count"),
    )
)
df.to_csv(out / "seed_metrics.csv", index=False)
summary.to_csv(out / "summary.csv", index=False)
print(summary.to_string(index=False))
PY
