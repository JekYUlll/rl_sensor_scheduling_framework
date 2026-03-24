#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.sequence_metrics import dtw_distance_1d, pearson_1d, smape_1d

MetricRecord = dict[str, Any]


def _discover_prediction_runs(run_tag: str, model: str | None) -> list[tuple[str, str, Path]]:
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
        if model is not None and model_name != model:
            continue
        pred_path = path / "forecast_predictions.npz"
        if pred_path.exists():
            discovered.append((scheduler_name, model_name, pred_path))
    return discovered


def _resolve_target_index(feature_names: list[str], target: str) -> int:
    if target.isdigit():
        idx = int(target)
        if idx < 0 or idx >= len(feature_names):
            raise ValueError(f"target index {idx} out of range [0, {len(feature_names) - 1}]")
        return idx
    if target not in feature_names:
        raise ValueError(f"target '{target}' not found in feature_names: {feature_names}")
    return feature_names.index(target)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _load_comparison_table(run_tag: str) -> pd.DataFrame:
    path = ROOT / "reports" / "aggregate" / f"metrics_forecast_all_{run_tag}_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _plot_overlay(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    title: str,
    out_path: Path,
    metrics_map: dict[str, MetricRecord],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(y_true))
    ax.plot(x, y_true, color="black", linewidth=2.0, label="true")
    for scheduler, series in preds.items():
        meta = metrics_map.get(scheduler, {})
        rmse_delta = _as_finite_float(meta.get("rmse_selected_target_increase_pct_vs_full_open"))
        power_save = _as_finite_float(meta.get("power_saving_pct_vs_full_open"))
        label = scheduler
        if rmse_delta is not None and power_save is not None:
            label = f"{scheduler} (dRMSE={rmse_delta:+.2f}%, save={power_save:.1f}%)"
        ax.plot(x, series, linewidth=1.4, alpha=0.9, label=label)
    ax.set_title(title)
    ax.set_xlabel("test window index")
    ax.set_ylabel("value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_small_multiples(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    title: str,
    out_path: Path,
    metrics_map: dict[str, MetricRecord],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schedulers = list(preds)
    n = len(schedulers)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    x = np.arange(len(y_true))

    for ax, scheduler in zip(axes_arr.flatten(), schedulers):
        series = preds[scheduler]
        meta = metrics_map.get(scheduler, {})
        rmse_delta = _as_finite_float(meta.get("rmse_selected_target_increase_pct_vs_full_open"))
        power_save = _as_finite_float(meta.get("power_saving_pct_vs_full_open"))
        ax.plot(x, y_true, color="black", linewidth=1.8, label="true")
        ax.plot(x, series, linewidth=1.2, alpha=0.95, label=scheduler)
        subtitle = scheduler
        if rmse_delta is not None and power_save is not None:
            subtitle = f"{scheduler} | dRMSE={rmse_delta:+.2f}%, save={power_save:.1f}%"
        ax.set_title(subtitle, fontsize=10)
        ax.grid(alpha=0.25)

    for ax in axes_arr.flatten()[n:]:
        ax.axis("off")

    handles, labels = axes_arr.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    fig.suptitle(title, y=1.08, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model", default="all", help="predictor name or 'all'")
    parser.add_argument("--target", default="snow_mass_flux_kg_m2_s", help="feature name or integer index")
    parser.add_argument("--horizon", type=int, default=1, help="1-based horizon index")
    parser.add_argument("--max-points", type=int, default=300)
    parser.add_argument("--schedulers", nargs="*", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    comparison_df = _load_comparison_table(args.run_tag)
    metrics_map_all: dict[tuple[str, str], MetricRecord] = {}
    if not comparison_df.empty:
        for _, row in comparison_df.iterrows():
            metrics_map_all[(str(row["model"]), str(row["scheduler"]))] = {
                str(k): v for k, v in row.to_dict().items()
            }

    discovered = _discover_prediction_runs(args.run_tag, None if args.model == "all" else args.model)
    if not discovered:
        raise FileNotFoundError(f"no forecast_predictions.npz found for run_tag={args.run_tag}, model={args.model}")

    if args.schedulers:
        allowed = set(args.schedulers)
        discovered = [item for item in discovered if item[0] in allowed]
        if not discovered:
            raise FileNotFoundError("no prediction runs matched the requested schedulers")

    grouped: dict[str, list[tuple[str, Path]]] = {}
    for scheduler, model_name, pred_path in discovered:
        grouped.setdefault(model_name, []).append((scheduler, pred_path))

    out_root = Path(args.out_dir) if args.out_dir else ROOT / "reports" / "aggregate" / f"posthoc_{args.run_tag}" / "prediction_curves"
    out_root.mkdir(parents=True, exist_ok=True)

    for model_name, items in sorted(grouped.items()):
        preds: dict[str, np.ndarray] = {}
        reference_true: np.ndarray | None = None
        feature_names: list[str] | None = None

        for scheduler, pred_path in sorted(items):
            data = np.load(pred_path, allow_pickle=True)
            feature_key = "target_feature_names" if "target_feature_names" in data else "feature_names"
            features = data[feature_key].tolist()
            current_feature_names = [str(name) for name in features]
            if data["y_true"].shape[2] != len(current_feature_names):
                raise ValueError(
                    f"y_true channel count {data['y_true'].shape[2]} does not match target features "
                    f"{len(current_feature_names)} for {pred_path}"
                )
            if data["y_pred"].shape[2] != len(current_feature_names):
                raise ValueError(
                    f"y_pred channel count {data['y_pred'].shape[2]} does not match target features "
                    f"{len(current_feature_names)} for {pred_path}; stale or invalid prediction artifact"
                )
            if feature_names is None:
                feature_names = current_feature_names
            elif feature_names != current_feature_names:
                raise ValueError(f"feature_names mismatch for {pred_path}")

            target_idx = _resolve_target_index(current_feature_names, str(args.target))
            horizon_idx = args.horizon - 1
            y_true = data["y_true"][:, horizon_idx, target_idx].astype(float)
            y_pred = data["y_pred"][:, horizon_idx, target_idx].astype(float)

            if reference_true is None:
                reference_true = y_true
            elif not np.allclose(reference_true, y_true, equal_nan=True):
                raise ValueError(f"y_true mismatch across schedulers for model={model_name}")

            preds[scheduler] = y_pred

        assert reference_true is not None
        assert feature_names is not None

        y_true_plot = reference_true[: args.max_points]
        preds_plot = {scheduler: series[: args.max_points] for scheduler, series in preds.items()}
        target_idx = _resolve_target_index(feature_names, str(args.target))
        target_name = feature_names[target_idx]

        metrics_rows = []
        metrics_map_model: dict[str, MetricRecord] = {}
        full_open_rmse = None
        full_open_dtw = None
        full_open_pearson = None
        if "full_open" in preds:
            full_open_rmse = _rmse(reference_true, preds["full_open"])
            full_open_dtw = dtw_distance_1d(reference_true, preds["full_open"])
            full_open_pearson = pearson_1d(reference_true, preds["full_open"])
        for scheduler, series in preds.items():
            metrics = metrics_map_all.get((model_name, scheduler), {})
            selected_rmse = _rmse(reference_true, series)
            selected_mae = _mae(reference_true, series)
            selected_smape = smape_1d(reference_true, series)
            selected_pearson = pearson_1d(reference_true, series)
            selected_dtw = dtw_distance_1d(reference_true, series)
            selected_delta = float("nan")
            selected_dtw_delta = float("nan")
            selected_pearson_delta = float("nan")
            if full_open_rmse is not None and np.isfinite(full_open_rmse) and full_open_rmse > 1e-12:
                selected_delta = 100.0 * (selected_rmse - full_open_rmse) / full_open_rmse
            if full_open_dtw is not None and np.isfinite(full_open_dtw) and full_open_dtw > 1e-12:
                selected_dtw_delta = 100.0 * (selected_dtw - full_open_dtw) / full_open_dtw
            if full_open_pearson is not None and np.isfinite(full_open_pearson):
                selected_pearson_delta = selected_pearson - full_open_pearson
            metrics_local = dict(metrics)
            metrics_local["rmse_selected_target"] = selected_rmse
            metrics_local["rmse_selected_target_increase_pct_vs_full_open"] = selected_delta
            metrics_local["mae_selected_target"] = selected_mae
            metrics_local["smape_selected_target"] = selected_smape
            metrics_local["pearson_selected_target"] = selected_pearson
            metrics_local["dtw_selected_target"] = selected_dtw
            metrics_local["dtw_selected_target_increase_pct_vs_full_open"] = selected_dtw_delta
            metrics_local["pearson_selected_target_delta_vs_full_open"] = selected_pearson_delta
            metrics_map_model[scheduler] = metrics_local
            metrics_rows.append(
                {
                    "model": model_name,
                    "scheduler": scheduler,
                    "target": target_name,
                    "horizon": args.horizon,
                    "rmse_selected_target": selected_rmse,
                    "rmse_selected_target_increase_pct_vs_full_open": selected_delta,
                    "mae_selected_target": selected_mae,
                    "smape_selected_target": selected_smape,
                    "pearson_selected_target": selected_pearson,
                    "pearson_selected_target_delta_vs_full_open": selected_pearson_delta,
                    "dtw_selected_target": selected_dtw,
                    "dtw_selected_target_increase_pct_vs_full_open": selected_dtw_delta,
                    "rmse_increase_pct_vs_full_open_all_features": metrics.get("rmse_increase_pct_vs_full_open", float("nan")),
                    "dtw_h1_increase_pct_vs_full_open_all_features": metrics.get("dtw_h1_increase_pct_vs_full_open", float("nan")),
                    "pearson_h1_delta_vs_full_open_all_features": metrics.get("pearson_h1_delta_vs_full_open", float("nan")),
                    "power_saving_pct_vs_full_open": metrics.get("power_saving_pct_vs_full_open", float("nan")),
                }
            )

        out_dir = out_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"{model_name} | target={target_name} | H={args.horizon}"
        _plot_overlay(
            y_true=y_true_plot,
            preds=preds_plot,
            title=title,
            out_path=out_dir / f"{target_name}_H{args.horizon}_overlay.png",
            metrics_map=metrics_map_model,
        )
        _plot_small_multiples(
            y_true=y_true_plot,
            preds=preds_plot,
            title=title,
            out_path=out_dir / f"{target_name}_H{args.horizon}_small_multiples.png",
            metrics_map=metrics_map_model,
        )
        pd.DataFrame(metrics_rows).sort_values("rmse_selected_target").to_csv(
            out_dir / f"{target_name}_H{args.horizon}_summary.csv",
            index=False,
        )
        print(f"[ok] wrote comparison plots for model={model_name} -> {out_dir}")


if __name__ == "__main__":
    main()
