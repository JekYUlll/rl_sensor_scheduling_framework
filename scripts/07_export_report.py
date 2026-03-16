#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    run_dir = Path("reports/runs") / args.run_id
    summary = {}
    est = run_dir / "metrics_estimation.csv"
    fc = run_dir / "metrics_forecast.csv"
    if est.exists():
        summary["estimation"] = pd.read_csv(est).to_dict(orient="records")
    if fc.exists():
        summary["forecast"] = pd.read_csv(fc).to_dict(orient="records")

    out_path = run_dir / "report_summary.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"[{k}]\n{v}\n")
    print(out_path)


if __name__ == "__main__":
    main()
