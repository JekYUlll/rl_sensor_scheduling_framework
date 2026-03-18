from __future__ import annotations

import numpy as np

from sensors.base_sensor import SensorSpec
from sensors.dataset_sensor import DatasetSensor


def test_dataset_sensor_returns_selection_matrix():
    spec = SensorSpec(
        sensor_id="wind",
        obs_dim=2,
        variables=["wind_speed_ms", "air_temperature_c"],
        refresh_interval=1,
        power_cost=1.0,
        noise_std=0.0,
    )
    sensor = DatasetSensor(spec=spec, state_columns=["wind_speed_ms", "relative_humidity", "air_temperature_c"])
    obs = sensor.observe(
        {
            "wind_speed_ms": 5.0,
            "relative_humidity": 70.0,
            "air_temperature_c": -12.0,
        },
        t=0,
    )
    assert obs["available"] is True
    assert np.allclose(obs["y"], np.array([5.0, -12.0]))
    assert obs["C"].shape == (2, 3)
    assert np.allclose(obs["C"], np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
