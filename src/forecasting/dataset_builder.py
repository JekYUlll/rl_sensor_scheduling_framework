from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastDataset:
    X: np.ndarray
    Y: np.ndarray
    target_indices: np.ndarray | None = None


def build_window_dataset(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    target_series: np.ndarray | None = None,
    target_indices: np.ndarray | None = None,
) -> ForecastDataset:
    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    if target_series is None:
        target_series = series
    else:
        target_series = np.asarray(target_series, dtype=float)
        if target_series.ndim == 1:
            target_series = target_series.reshape(-1, 1)
    if series.shape[0] != target_series.shape[0]:
        raise ValueError("series and target_series must share the same time dimension")
    n = series.shape[0]
    xs = []
    ys = []
    for i in range(lookback, n - horizon + 1):
        xs.append(series[i - lookback : i])
        ys.append(target_series[i : i + horizon])
    if not xs:
        return ForecastDataset(
            X=np.empty((0, lookback, series.shape[1])),
            Y=np.empty((0, horizon, target_series.shape[1])),
            target_indices=None if target_indices is None else np.asarray(target_indices, dtype=int),
        )
    return ForecastDataset(
        X=np.asarray(xs),
        Y=np.asarray(ys),
        target_indices=None if target_indices is None else np.asarray(target_indices, dtype=int),
    )


def split_dataset(ds: ForecastDataset, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n = ds.X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = ForecastDataset(ds.X[:n_train], ds.Y[:n_train], ds.target_indices)
    val = ForecastDataset(ds.X[n_train : n_train + n_val], ds.Y[n_train : n_train + n_val], ds.target_indices)
    test = ForecastDataset(ds.X[n_train + n_val :], ds.Y[n_train + n_val :], ds.target_indices)
    return train, val, test
