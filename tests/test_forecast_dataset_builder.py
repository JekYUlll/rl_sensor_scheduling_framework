from __future__ import annotations

import numpy as np

from forecasting.dataset_builder import build_window_dataset


def test_dataset_builder_alignment():
    arr = np.arange(20, dtype=float).reshape(-1, 1)
    ds = build_window_dataset(arr, lookback=5, horizon=2)
    assert ds.X.shape[0] == 14
    assert ds.X.shape[1] == 5
    assert ds.Y.shape[1] == 2
    # first label should start at index 5
    assert ds.Y[0, 0, 0] == 5.0


def test_dataset_builder_supports_separate_target_series():
    source = np.arange(20, dtype=float).reshape(-1, 1)
    target = (np.arange(20, dtype=float) * 10.0).reshape(-1, 1)
    ds = build_window_dataset(source, lookback=4, horizon=3, target_series=target)
    assert ds.X.shape[0] == 14
    assert ds.Y[0, 0, 0] == 40.0
