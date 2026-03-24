from __future__ import annotations

import numpy as np

from evaluation.sequence_metrics import dtw_distance_1d, pearson_1d, smape_1d


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    smape = float(smape_1d(y_true, y_pred))

    pearsons: list[float] = []
    dtws: list[float] = []
    if y_true.ndim == 3:
        for target_idx in range(y_true.shape[2]):
            seq_true = y_true[:, 0, target_idx]
            seq_pred = y_pred[:, 0, target_idx]
            pearsons.append(pearson_1d(seq_true, seq_pred))
            dtws.append(dtw_distance_1d(seq_true, seq_pred))
    else:
        pearsons.append(pearson_1d(y_true, y_pred))
        dtws.append(dtw_distance_1d(y_true, y_pred))

    pearson_mean = float(np.nanmean(pearsons)) if pearsons else float("nan")
    dtw_mean = float(np.nanmean(dtws)) if dtws else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "pearson_h1_mean": pearson_mean,
        "dtw_h1_mean": dtw_mean,
    }
