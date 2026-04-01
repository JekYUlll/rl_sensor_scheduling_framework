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
from forecasting.series_preparation import prepare_input_and_targets


def _normalize_split(train: ForecastDataset, val: ForecastDataset, test: ForecastDataset):
    x_mean = train.X.mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(train.X.std(axis=(0, 1), keepdims=True), 1e-6)
    y_mean = train.Y.mean(axis=(0, 1), keepdims=True)
    y_std = np.maximum(train.Y.std(axis=(0, 1), keepdims=True), 1e-6)

    def _apply(ds: ForecastDataset) -> ForecastDataset:
        return ForecastDataset(
            X=(ds.X - x_mean) / x_std,
            Y=(ds.Y - y_mean) / y_std,
            target_indices=ds.target_indices,
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
    base_feature_names = data["feature_names"].tolist() if "feature_names" in data else [f"f{i}" for i in range(input_series.shape[1])]
    observed_mask = data["observed_mask"] if "observed_mask" in data else None
    time_index = data["time_index"] if "time_index" in data else None
    meta = _load_dataset_meta(args.series_npz)
    cfg = load_yaml(args.predictor_cfg)
    use_observed_mask = bool(cfg.get("use_observed_mask", False))
    use_time_delta = bool(cfg.get("use_time_delta", False))
    target_columns = [str(name) for name in meta.get("forecast_target_columns", meta.get("reward_target_columns", []))]
    input_series, input_feature_names, target_series, target_feature_names, target_indices = prepare_input_and_targets(
        input_series=np.asarray(input_series, dtype=float),
        target_series=np.asarray(raw_target_series, dtype=float),
        feature_names=[str(name) for name in base_feature_names],
        observed_mask=None if observed_mask is None else np.asarray(observed_mask, dtype=float),
        use_observed_mask=use_observed_mask,
        use_time_delta=use_time_delta,
        target_columns=target_columns,
        time_index=None if time_index is None else np.asarray(time_index, dtype=int),
        base_freq_s=int(meta.get("base_freq_s", 1)),
    )

    ds = build_window_dataset(
        series=input_series,
        lookback=args.lookback,
        horizon=args.horizon,
        target_series=target_series,
        target_indices=target_indices,
    )
    train, val, test = split_dataset(ds)
    train_norm, val_norm, test_norm, stats = _normalize_split(train, val, test)

    if args.device:
        cfg["device"] = args.device
    predictor = build_predictor(cfg)
    if hasattr(predictor, "set_context"):
        predictor.set_context(
            input_feature_names=list(input_feature_names),
            target_feature_names=list(target_feature_names),
            stats=stats,
        )
    predictor.fit(train_norm, val_norm)
    pred_norm = predictor.predict(test_norm)
    if pred_norm.shape != test_norm.Y.shape:
        raise ValueError(
            "predictor output shape mismatch: "
            f"pred={pred_norm.shape}, target={test_norm.Y.shape}, "
            f"model={cfg.get('predictor_name', 'unknown')}"
        )
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
    history = getattr(predictor, "history", None)
    if isinstance(history, dict) and history:
        with (out_dir / "training_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
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
