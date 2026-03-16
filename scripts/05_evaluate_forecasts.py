#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", default="reports/runs")
    parser.add_argument("--out_csv", default="reports/aggregate/metrics_forecast_all.csv")
    args = parser.parse_args()

    root = Path(args.reports_dir)
    rows = []
    for p in root.glob("*/metrics_forecast.csv"):
        df = pd.read_csv(p)
        df["run_id"] = p.parent.name
        rows.append(df)
    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame()
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
