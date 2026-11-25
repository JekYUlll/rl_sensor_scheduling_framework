from __future__ import annotations

import numpy as np

from sensors.base_sensor import SensorSpec
from sensors.linear_sensor import LinearSensor


def test_sensor_refresh_interval():
    spec = SensorSpec(sensor_id="s", obs_dim=1, variables=["x0"], refresh_interval=2, power_cost=1.0, noise_std=0.1)
    sensor = LinearSensor(spec=spec, c_vec=np.array([1.0]), r_diag=np.array([0.1]))
    x = np.array([0.0])
    out1 = sensor.observe(x, t=0)
    out2 = sensor.observe(x, t=1)
    out3 = sensor.observe(x, t=2)
    assert out1["available"] is True
    assert out2["available"] is False
    assert out3["available"] is True
