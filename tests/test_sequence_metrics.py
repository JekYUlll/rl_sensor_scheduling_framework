from __future__ import annotations

import numpy as np

from evaluation.forecast_metrics import compute_forecast_metrics
from evaluation.sequence_metrics import dtw_distance_1d


def test_dtw_distance_1d_matches_exact_for_short_inputs() -> None:
    y_true = np.asarray([0.0, 1.0, 2.0, 3.0, 2.0], dtype=float)
    y_pred = np.asarray([0.0, 1.0, 1.5, 2.5, 2.0], dtype=float)

    exact = dtw_distance_1d(y_true, y_pred, max_points=None)
    bounded = dtw_distance_1d(y_true, y_pred)

    assert np.isclose(bounded, exact, atol=1e-12)


def test_dtw_distance_1d_downsamples_long_inputs_but_stays_exact_for_identical_signal() -> None:
    base = np.sin(np.linspace(0.0, 24.0 * np.pi, num=5000, dtype=float))

    dist = dtw_distance_1d(base, base.copy())

    assert np.isclose(dist, 0.0, atol=1e-12)


def test_compute_forecast_metrics_handles_long_route_a_style_sequences() -> None:
    steps = 5000
    targets = 3
    horizons = 3
    phase = np.linspace(0.0, 8.0 * np.pi, num=steps, dtype=float)
    y_true = np.zeros((steps, horizons, targets), dtype=float)
    y_pred = np.zeros_like(y_true)
    for horizon_idx in range(horizons):
        for target_idx in range(targets):
            signal = np.sin(phase + 0.1 * horizon_idx + 0.2 * target_idx)
            y_true[:, horizon_idx, target_idx] = signal
            y_pred[:, horizon_idx, target_idx] = signal + 0.01 * (target_idx + 1)

    metrics = compute_forecast_metrics(y_true, y_pred)

    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["smape"])
    assert np.isfinite(metrics["pearson_h1_mean"])
    assert np.isfinite(metrics["dtw_h1_mean"])
