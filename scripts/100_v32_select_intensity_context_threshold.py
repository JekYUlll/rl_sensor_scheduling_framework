#!/usr/bin/env python
"""Select an intensity-context threshold using calibration ledgers only."""
from __future__ import annotations

import argparse
import glob
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_baseline_module():
    path = ROOT / "scripts" / "81_v31_framework_baseline_supplements.py"
    spec = importlib.util.spec_from_file_location("framework_baselines", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-glob", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    candidates: list[dict[str, object]] = []
    for metrics_path_str in sorted(glob.glob(str(args.candidate_glob))):
        metrics_path = Path(metrics_path_str)
        seed_dir = metrics_path.parent
        ledger_path = seed_dir / "context_replay_calibration_ledger.csv"
        action_path = seed_dir / "context_bandit_action_map.csv"
        if not ledger_path.is_file() or not action_path.is_file():
            raise FileNotFoundError(f"Incomplete threshold candidate: {seed_dir}")
        metrics = pd.read_csv(metrics_path)
        ledger = pd.read_csv(ledger_path)
        actions = pd.read_csv(action_path)
        if len(metrics) != 1 or len(actions) != 1:
            raise ValueError(f"Expected one policy row in {seed_dir}")
        seed = int(metrics.iloc[0]["seed"])
        threshold = float(actions.iloc[0]["high_threshold"])
        candidates.append({
            "seed": seed,
            "high_threshold": threshold,
            "selection_primary": float(ledger["selection_primary"].min()),
            "selection_secondary": float(
                ledger.loc[ledger["selection_primary"].idxmin(), "selection_secondary"]
            ),
            "metrics_path": metrics_path,
            "ledger_path": ledger_path,
            "action_path": action_path,
        })

    candidate_table = pd.DataFrame(candidates).sort_values(
        ["seed", "selection_primary", "selection_secondary", "high_threshold"]
    )
    if candidate_table.empty:
        raise SystemExit("No threshold candidates found")
    selected = candidate_table.groupby("seed", as_index=False).first()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    candidate_table.to_csv(out_root / "threshold_candidate_ledger.csv", index=False)
    selected.to_csv(out_root / "selected_thresholds.csv", index=False)

    rows: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        metrics = pd.read_csv(row.metrics_path)
        metrics["calibration_selected_high_threshold"] = float(row.high_threshold)
        metrics["calibration_selection_primary"] = float(row.selection_primary)
        metrics["calibration_selection_secondary"] = float(row.selection_secondary)
        rows.append(metrics)
    result = pd.concat(rows, ignore_index=True)
    baseline = load_baseline_module()
    baseline.aggregate(result, out_root)


if __name__ == "__main__":
    main()
