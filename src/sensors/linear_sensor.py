from __future__ import annotations

import numpy as np

from sensors.base_sensor import AbstractSensor, SensorSpec


class LinearSensor(AbstractSensor):
    def __init__(self, spec: SensorSpec, c_vec: np.ndarray, r_diag: np.ndarray) -> None:
        super().__init__(spec)
        self.c_vec = np.asarray(c_vec, dtype=float).reshape(1, -1)
        self.r_diag = np.asarray(r_diag, dtype=float).reshape(-1)

    def observe(self, latent_state: np.ndarray, t: int | None = None) -> dict:
        if t is None:
            raise ValueError("LinearSensor.observe requires time index t")
        self._ensure_legacy_direct_observe_ready()
        if not self.can_sample(t):
            return {"sensor_id": self.sensor_id, "available": False}
        clean = (self.c_vec @ latent_state.reshape(-1, 1)).reshape(-1)
        noise = np.random.randn(clean.shape[0]) * np.sqrt(self.r_diag)
        y = clean + noise
        self._touch_sampled(t)
        return {
            "sensor_id": self.sensor_id,
            "available": True,
            "y": y,
            "C": self.c_vec,
            "R": np.diag(self.r_diag),
            "power_cost": self.power_cost(),
        }
