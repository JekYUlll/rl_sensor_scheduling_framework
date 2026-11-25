from __future__ import annotations

import numpy as np

from estimators.kalman_filter import KalmanFilterEstimator
from estimators.observation_preprocessor import ObservationPreprocessor


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


def test_observation_preprocessor_smooths_and_clips_outliers():
    pre = ObservationPreprocessor.from_config(
        {
            "enabled": True,
            "alpha": 0.5,
            "clip_sigma": 2.0,
            "effective_noise_scale": None,
        }
    )
    obs0 = {
        "available": True,
        "sensor_id": "s0",
        "variables": ["x"],
        "y": np.asarray([10.0]),
        "R": np.asarray([[1.0]]),
        "t": 0,
    }
    obs1 = {
        "available": True,
        "sensor_id": "s0",
        "variables": ["x"],
        "y": np.asarray([100.0]),
        "R": np.asarray([[1.0]]),
        "t": 1,
    }

    first = pre.process([obs0])[0]
    second = pre.process([obs1])[0]

    np.testing.assert_allclose(first["y"], [10.0])
    np.testing.assert_allclose(second["y"], [11.0])
    np.testing.assert_allclose(second["y_raw"], [100.0])


def test_kf_update_uses_observation_preprocessor():
    pre = ObservationPreprocessor.from_config(
        {
            "enabled": True,
            "alpha": 0.5,
            "clip_sigma": 2.0,
            "effective_noise_scale": None,
        }
    )
    kf = KalmanFilterEstimator(
        np.eye(1),
        np.zeros((1, 1)),
        np.asarray([0.0]),
        np.eye(1) * 0.01,
        sensor_ids=["s0"],
        observation_preprocessor=pre,
    )
    obs = {
        "available": True,
        "sensor_id": "s0",
        "variables": ["x"],
        "C": np.asarray([[1.0]]),
        "R": np.asarray([[1.0]]),
    }
    kf.update([{**obs, "y": np.asarray([10.0]), "t": 0}])
    first_estimate = float(kf.x_hat[0])
    kf.update([{**obs, "y": np.asarray([100.0]), "t": 1}])
    assert float(kf.x_hat[0]) < 1.0 + first_estimate
