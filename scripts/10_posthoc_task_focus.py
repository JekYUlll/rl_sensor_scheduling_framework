#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import load_yaml
from evaluation.sequence_metrics import dtw_distance_1d, pearson_1d, smape_1d
from visualization.posthoc_plots import heatmap


def _discover_prediction_runs(run_tag: str) -> list[tuple[str, str, Path]]:
    runs_dir = ROOT / "reports" / "runs"
    discovered: list[tuple[str, str, Path]] = []
    pattern = re.compile(rf"^{re.escape(run_tag)}_(.+)_pred_(.+)$")
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        scheduler_name, model_name = match.groups()
        pred_path = path / "forecast_predictions.npz"
        if pred_path.exists():
            discovered.append((scheduler_name, model_name, pred_path))
    return discovered


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _load_primary_targets(env_cfg: Path) -> list[str]:
    cfg = load_yaml(str(env_cfg))
    return [str(v) for v in cfg.get("reward_target_columns", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task-focused posthoc summaries for primary microclimate targets")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--env-cfg", default="configs/env/windblown_case.yaml")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / "aggregate" / f"posthoc_{args.run_tag}" / "task_focus_primary"
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_targets = _load_primary_targets(ROOT / args.env_cfg)
    discovered = _discover_prediction_runs(args.run_tag)
    if not discovered:
        raise FileNotFoundError(f"no forecast predictions found for run_tag={args.run_tag}")

    rows: list[dict[str, object]] = []
    by_model_target_h: dict[tuple[str, str, int], dict[str, dict[str, float]]] = {}

    for scheduler, model_name, pred_path in discovered:
        data = np.load(pred_path, allow_pickle=True)
        feature_names = [str(v) for v in data["target_feature_names"].tolist()]
        y_true = np.asarray(data["y_true"], dtype=float)
        y_pred = np.asarray(data["y_pred"], dtype=float)
        for target in primary_targets:
            if target not in feature_names:
                continue
            target_idx = feature_names.index(target)
            for horizon_idx in range(y_true.shape[1]):
                seq_true = y_true[:, horizon_idx, target_idx]
                seq_pred = y_pred[:, horizon_idx, target_idx]
                key = (model_name, target, horizon_idx + 1)
                by_model_target_h.setdefault(key, {})[scheduler] = {
                    "rmse": _rmse(seq_true, seq_pred),
                    "mae": _mae(seq_true, seq_pred),
                    "smape": smape_1d(seq_true, seq_pred),
                    "pearson": pearson_1d(seq_true, seq_pred),
                    "dtw": dtw_distance_1d(seq_true, seq_pred),
                }

    for (model_name, target, horizon), scheduler_map in sorted(by_model_target_h.items()):
        baseline = scheduler_map.get("full_open")
        if baseline is None:
            continue
        for scheduler, metrics in scheduler_map.items():
            row = {
                "model": model_name,
                "target": target,
                "horizon": horizon,
                "scheduler": scheduler,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "smape": metrics["smape"],
                "pearson": metrics["pearson"],
                "dtw": metrics["dtw"],
                "rmse_increase_pct_vs_full_open": 100.0 * (metrics["rmse"] - baseline["rmse"]) / max(baseline["rmse"], 1e-12),
                "dtw_increase_pct_vs_full_open": 100.0 * (metrics["dtw"] - baseline["dtw"]) / max(baseline["dtw"], 1e-12),
                "pearson_delta_vs_full_open": metrics["pearson"] - baseline["pearson"],
            }
            rows.append(row)

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("task-focused posthoc produced no rows")
    long_df.to_csv(out_dir / "primary_target_metrics_long.csv", index=False)

    summary = (
        long_df.groupby("scheduler", dropna=False)
        .agg(
            rmse_increase_pct_vs_full_open=("rmse_increase_pct_vs_full_open", "mean"),
            dtw_increase_pct_vs_full_open=("dtw_increase_pct_vs_full_open", "mean"),
            pearson_delta_vs_full_open=("pearson_delta_vs_full_open", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "primary_target_scheduler_summary.csv", index=False)

    for horizon in sorted(long_df["horizon"].unique()):
        subset = long_df[long_df["horizon"] == horizon].copy()
        rmse_pivot = subset.pivot_table(index="scheduler", columns="target", values="rmse_increase_pct_vs_full_open", aggfunc="mean")
        rmse_pivot.to_csv(out_dir / f"primary_target_rmse_increase_h{horizon}.csv")
        heatmap(rmse_pivot, out_dir / f"primary_target_rmse_increase_h{horizon}.png", title=f"Primary-target RMSE increase vs full-open (H={horizon})", vmin=None, vmax=None)

        dtw_pivot = subset.pivot_table(index="scheduler", columns="target", values="dtw_increase_pct_vs_full_open", aggfunc="mean")
        dtw_pivot.to_csv(out_dir / f"primary_target_dtw_increase_h{horizon}.csv")
        heatmap(dtw_pivot, out_dir / f"primary_target_dtw_increase_h{horizon}.png", title=f"Primary-target DTW increase vs full-open (H={horizon})", vmin=None, vmax=None)

        pearson_pivot = subset.pivot_table(index="scheduler", columns="target", values="pearson_delta_vs_full_open", aggfunc="mean")
        pearson_pivot.to_csv(out_dir / f"primary_target_pearson_delta_h{horizon}.csv")
        heatmap(pearson_pivot, out_dir / f"primary_target_pearson_delta_h{horizon}.png", title=f"Primary-target Pearson delta vs full-open (H={horizon})", vmin=None, vmax=None)

    print(out_dir / "primary_target_scheduler_summary.csv")


if __name__ == "__main__":
    main()
