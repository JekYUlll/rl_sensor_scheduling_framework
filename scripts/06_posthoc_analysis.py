#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.rank_analysis import rank_correlation
from visualization.posthoc_plots import heatmap


def _plot_power_vs_rmse(comp: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for scheduler, group in comp.groupby("scheduler"):
        ax.scatter(
            group["power_saving_pct_vs_full_open"],
            group["rmse_increase_pct_vs_full_open"],
            label=scheduler,
            s=64,
            alpha=0.85,
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Power saving vs full-open (%)")
    ax.set_ylabel("RMSE increase vs full-open (%)")
    ax.set_title("Prediction degradation vs power saving")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_power_vs_metric(
    comp: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for scheduler, group in comp.groupby("scheduler"):
        ax.scatter(
            group[x_col],
            group[y_col],
            label=scheduler,
            s=64,
            alpha=0.85,
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Power saving vs full-open (%)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", default="reports/aggregate/metrics_forecast_all.csv")
    parser.add_argument("--out_dir", default="reports/aggregate")
    args = parser.parse_args()

    metrics_df = pd.read_csv(args.metrics_csv)
    if metrics_df.empty:
        raise RuntimeError("No metrics rows found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "scheduler" in metrics_df.columns and "model" in metrics_df.columns:
        corr = rank_correlation(metrics_df.rename(columns={"scheduler": "strategy"}), metric_col="rmse")
        corr.to_csv(out_dir / "rank_correlation.csv")
        heatmap(corr, out_dir / "rank_correlation.png", title="Rank correlation")

        rmse_pivot = metrics_df.pivot_table(index="scheduler", columns="model", values="rmse", aggfunc="mean")
        rmse_pivot.to_csv(out_dir / "scheduler_model_rmse.csv")
        heatmap(rmse_pivot, out_dir / "scheduler_model_rmse.png", title="Scheduler vs model RMSE", vmin=None, vmax=None)

    comp_path = Path(args.metrics_csv).with_name(f"{Path(args.metrics_csv).stem}_comparison.csv")
    if not comp_path.exists():
        print(out_dir / "rank_correlation.csv")
        return

    comp_df = pd.read_csv(comp_path)
    if comp_df.empty:
        print(out_dir / "rank_correlation.csv")
        return

    increase_pivot = comp_df.pivot_table(
        index="scheduler",
        columns="model",
        values="rmse_increase_pct_vs_full_open",
        aggfunc="mean",
    )
    increase_pivot.to_csv(out_dir / "scheduler_model_rmse_increase_vs_full_open.csv")
    heatmap(
        increase_pivot,
        out_dir / "scheduler_model_rmse_increase_vs_full_open.png",
        title="RMSE increase vs full-open (%)",
        vmin=None,
        vmax=None,
    )

    dtw_pivot = comp_df.pivot_table(
        index="scheduler",
        columns="model",
        values="dtw_h1_increase_pct_vs_full_open",
        aggfunc="mean",
    )
    dtw_pivot.to_csv(out_dir / "scheduler_model_dtw_h1_increase_vs_full_open.csv")
    heatmap(
        dtw_pivot,
        out_dir / "scheduler_model_dtw_h1_increase_vs_full_open.png",
        title="DTW(H=1) increase vs full-open (%)",
        vmin=None,
        vmax=None,
    )

    pearson_pivot = comp_df.pivot_table(
        index="scheduler",
        columns="model",
        values="pearson_h1_delta_vs_full_open",
        aggfunc="mean",
    )
    pearson_pivot.to_csv(out_dir / "scheduler_model_pearson_h1_delta_vs_full_open.csv")
    heatmap(
        pearson_pivot,
        out_dir / "scheduler_model_pearson_h1_delta_vs_full_open.png",
        title="Pearson(H=1) delta vs full-open",
        vmin=None,
        vmax=None,
    )

    _plot_power_vs_rmse(comp_df, out_dir / "power_saving_vs_rmse_increase.png")
    _plot_power_vs_metric(
        comp_df,
        x_col="power_saving_pct_vs_full_open",
        y_col="dtw_h1_increase_pct_vs_full_open",
        out_path=out_dir / "power_saving_vs_dtw_h1_increase.png",
        title="Prediction DTW(H=1) degradation vs power saving",
        y_label="DTW(H=1) increase vs full-open (%)",
    )
    _plot_power_vs_metric(
        comp_df,
        x_col="power_saving_pct_vs_full_open",
        y_col="pearson_h1_delta_vs_full_open",
        out_path=out_dir / "power_saving_vs_pearson_h1_delta.png",
        title="Prediction Pearson(H=1) delta vs power saving",
        y_label="Pearson(H=1) delta vs full-open",
    )
    scheduler_summary = (
        comp_df.groupby("scheduler", dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_increase_pct_vs_full_open=("rmse_increase_pct_vs_full_open", "mean"),
            dtw_h1_increase_pct_vs_full_open=("dtw_h1_increase_pct_vs_full_open", "mean"),
            pearson_h1_delta_vs_full_open=("pearson_h1_delta_vs_full_open", "mean"),
            power_saving_pct_vs_full_open=("power_saving_pct_vs_full_open", "mean"),
        )
        .reset_index()
    )
    scheduler_summary.to_csv(out_dir / "scheduler_summary.csv", index=False)
    print(out_dir / "scheduler_summary.csv")


if __name__ == "__main__":
    main()
