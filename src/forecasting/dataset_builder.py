from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastDataset:
    X: np.ndarray
    Y: np.ndarray


def build_window_dataset(series: np.ndarray, lookback: int, horizon: int) -> ForecastDataset:
    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    n = series.shape[0]
    xs = []
    ys = []
    for i in range(lookback, n - horizon + 1):
        xs.append(series[i - lookback : i])
        ys.append(series[i : i + horizon])
    if not xs:
        return ForecastDataset(X=np.empty((0, lookback, series.shape[1])), Y=np.empty((0, horizon, series.shape[1])))
    return ForecastDataset(X=np.asarray(xs), Y=np.asarray(ys))


def split_dataset(ds: ForecastDataset, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n = ds.X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = ForecastDataset(ds.X[:n_train], ds.Y[:n_train])
    val = ForecastDataset(ds.X[n_train : n_train + n_val], ds.Y[n_train : n_train + n_val])
    test = ForecastDataset(ds.X[n_train + n_val :], ds.Y[n_train + n_val :])
    return train, val, test
