#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import load_yaml
from evaluation.forecast_metrics import compute_forecast_metrics
from forecasting.dataset_builder import ForecastDataset, build_window_dataset, split_dataset
from forecasting.factory import build_predictor


def _normalize_split(train: ForecastDataset, val: ForecastDataset, test: ForecastDataset):
    x_mean = train.X.mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(train.X.std(axis=(0, 1), keepdims=True), 1e-6)
    y_mean = train.Y.mean(axis=(0, 1), keepdims=True)
    y_std = np.maximum(train.Y.std(axis=(0, 1), keepdims=True), 1e-6)

    def _apply(ds: ForecastDataset) -> ForecastDataset:
        return ForecastDataset(
            X=(ds.X - x_mean) / x_std,
            Y=(ds.Y - y_mean) / y_std,
        )

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return _apply(train), _apply(val), _apply(test), stats


def _load_dataset_meta(series_npz: str) -> dict:
    meta_path = Path(series_npz).with_suffix(".meta.yaml")
    if meta_path.exists():
        return load_yaml(meta_path)
    return {}


def _select_target_columns(
    target_series: np.ndarray,
    input_feature_names: list[str],
    meta: dict,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    configured = meta.get("reward_target_columns", [])
    if not configured:
        return np.asarray(target_series, dtype=float), list(input_feature_names), None
    target_names = [str(name) for name in configured if str(name) in input_feature_names]
    if not target_names:
        return np.asarray(target_series, dtype=float), list(input_feature_names), None
    indices = [input_feature_names.index(name) for name in target_names]
    return np.asarray(target_series, dtype=float)[:, indices], target_names, np.asarray(indices, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series_npz", required=True)
    parser.add_argument("--predictor_cfg", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data = np.load(args.series_npz, allow_pickle=True)
    input_series = data["input_series"] if "input_series" in data else data["series"]
    raw_target_series = data["target_series"] if "target_series" in data else input_series
    input_feature_names = data["feature_names"].tolist() if "feature_names" in data else [f"f{i}" for i in range(input_series.shape[1])]
    meta = _load_dataset_meta(args.series_npz)
    target_series, target_feature_names, target_indices = _select_target_columns(raw_target_series, list(input_feature_names), meta)

    ds = build_window_dataset(
        series=input_series,
        lookback=args.lookback,
        horizon=args.horizon,
        target_series=target_series,
        target_indices=target_indices,
    )
    train, val, test = split_dataset(ds)
    train_norm, val_norm, test_norm, stats = _normalize_split(train, val, test)

    cfg = load_yaml(args.predictor_cfg)
    if args.device:
        cfg["device"] = args.device
    predictor = build_predictor(cfg)
    predictor.fit(train_norm, val_norm)
    pred_norm = predictor.predict(test_norm)
    y_pred = pred_norm * stats["y_std"] + stats["y_mean"]

    metrics = compute_forecast_metrics(test.Y, y_pred)
    metrics_norm = compute_forecast_metrics(test_norm.Y, pred_norm)

    out_dir = ROOT / "reports" / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "model": cfg.get("predictor_name", "unknown"),
        "scheduler": meta.get("scheduler_name", "unknown"),
        "source_run_id": meta.get("run_id", ""),
        "dataset_npz": args.series_npz,
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "n_features": int(len(input_feature_names)),
        "n_targets": int(len(target_feature_names)),
        "device": str(cfg.get("device", "auto")),
        "target_columns": json.dumps(target_feature_names, ensure_ascii=False),
        "avg_power": float(meta.get("avg_power", float("nan"))),
        "total_power": float(meta.get("total_power", float("nan"))),
        "coverage_mean": float(meta.get("coverage_mean", float("nan"))),
        "trace_P_mean": float(meta.get("trace_P_mean", float("nan"))),
        "uncertainty_mean": float(meta.get("uncertainty_mean", float("nan"))),
        "full_open_power": float(meta.get("full_open_power", float("nan"))),
        **metrics,
        "rmse_norm": float(metrics_norm["rmse"]),
        "mae_norm": float(metrics_norm["mae"]),
        "mape_norm": float(metrics_norm["mape"]),
    }
    pd.DataFrame([row]).to_csv(out_dir / "metrics_forecast.csv", index=False)
    np.savez(
        out_dir / "forecast_predictions.npz",
        y_true=test.Y,
        y_pred=y_pred,
        y_true_norm=test_norm.Y,
        y_pred_norm=pred_norm,
        feature_names=np.asarray(target_feature_names),
        input_feature_names=np.asarray(input_feature_names),
        target_feature_names=np.asarray(target_feature_names),
    )
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
