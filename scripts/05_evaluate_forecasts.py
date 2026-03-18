#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _parse_run_tag(run_id: str) -> str:
    parts = run_id.split("_pred_")
    if len(parts) == 2:
        return parts[0]
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", default="reports/runs")
    parser.add_argument("--out_csv", default="reports/aggregate/metrics_forecast_all.csv")
    parser.add_argument("--run_tag", default=None)
    args = parser.parse_args()

    root = Path(args.reports_dir)
    rows = []
    for metrics_path in root.glob("*/metrics_forecast.csv"):
        run_id = metrics_path.parent.name
        if args.run_tag and (not run_id.startswith(args.run_tag)):
            continue
        df = pd.read_csv(metrics_path)
        df["run_id"] = run_id
        df["run_tag"] = _parse_run_tag(run_id)
        rows.append(df)

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if out.empty or "scheduler" not in out.columns or "model" not in out.columns:
        print(out_path)
        return

    baseline = out[out["scheduler"] == "full_open"].copy()
    if baseline.empty:
        print(out_path)
        return

    baseline = baseline[
        [
            "model",
            "rmse",
            "mae",
            "mape",
            "rmse_norm",
            "mae_norm",
            "avg_power",
            "total_power",
        ]
    ].rename(
        columns={
            "rmse": "rmse_full_open",
            "mae": "mae_full_open",
            "mape": "mape_full_open",
            "rmse_norm": "rmse_norm_full_open",
            "mae_norm": "mae_norm_full_open",
            "avg_power": "avg_power_full_open",
            "total_power": "total_power_full_open",
        }
    )
    comp = out.merge(baseline, on="model", how="left")
    comp["rmse_increase_pct_vs_full_open"] = 100.0 * (comp["rmse"] - comp["rmse_full_open"]) / np.maximum(comp["rmse_full_open"], 1e-12)
    comp["mae_increase_pct_vs_full_open"] = 100.0 * (comp["mae"] - comp["mae_full_open"]) / np.maximum(comp["mae_full_open"], 1e-12)
    comp["rmse_norm_increase_pct_vs_full_open"] = 100.0 * (comp["rmse_norm"] - comp["rmse_norm_full_open"]) / np.maximum(comp["rmse_norm_full_open"], 1e-12)
    comp["power_saving_pct_vs_full_open"] = 100.0 * (1.0 - comp["avg_power"] / np.maximum(comp["avg_power_full_open"], 1e-12))
    comp["total_energy_saving_pct_vs_full_open"] = 100.0 * (1.0 - comp["total_power"] / np.maximum(comp["total_power_full_open"], 1e-12))
    comp["rmse_per_unit_power"] = comp["rmse"] / np.maximum(comp["avg_power"], 1e-12)

    comp_path = out_path.with_name(f"{out_path.stem}_comparison.csv")
    comp.to_csv(comp_path, index=False)

    scheduler_summary = (
        comp.groupby("scheduler", dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_norm_mean=("rmse_norm", "mean"),
            rmse_increase_pct_vs_full_open=("rmse_increase_pct_vs_full_open", "mean"),
            power_saving_pct_vs_full_open=("power_saving_pct_vs_full_open", "mean"),
            total_energy_saving_pct_vs_full_open=("total_energy_saving_pct_vs_full_open", "mean"),
        )
        .reset_index()
    )
    scheduler_summary.to_csv(out_path.with_name(f"{out_path.stem}_scheduler_summary.csv"), index=False)
    print(out_path)


if __name__ == "__main__":
    main()
