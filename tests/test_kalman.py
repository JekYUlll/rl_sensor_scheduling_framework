from __future__ import annotations

import numpy as np

from estimators.kalman_filter import KalmanFilterEstimator


def test_kf_update_reduces_variance():
    A = np.array([[1.0]])
    Q = np.array([[0.01]])
    x0 = np.array([0.0])
    P0 = np.array([[1.0]])
    kf = KalmanFilterEstimator(A, Q, x0, P0, sensor_ids=["s0"])
    kf.predict()
    p_before = float(kf.P[0, 0])
    kf.update([{"available": True, "y": np.array([0.2]), "C": np.array([[1.0]]), "R": np.array([[0.1]])}])
    p_after = float(kf.P[0, 0])
    assert p_after < p_before
