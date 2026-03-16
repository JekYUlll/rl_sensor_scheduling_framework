#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.rank_analysis import rank_correlation
from visualization.posthoc_plots import heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", default="reports/aggregate/metrics_forecast_all.csv")
    parser.add_argument("--out_dir", default="reports/aggregate")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    if df.empty:
        raise RuntimeError("No metrics rows found")

    if "strategy" not in df.columns:
        df["strategy"] = df.get("run_id", "unknown")
    if "model" not in df.columns:
        df["model"] = "model"

    corr = rank_correlation(df, metric_col="rmse")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_dir / "rank_correlation.csv")
    heatmap(corr, out_dir / "rank_correlation.png", title="Rank correlation")
    print(out_dir / "rank_correlation.csv")


if __name__ == "__main__":
    main()
