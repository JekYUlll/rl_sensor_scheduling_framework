from __future__ import annotations

import numpy as np

from sensors.base_sensor import AbstractSensor, SensorSpec


class WindblownSensor(AbstractSensor):
    def observe(self, latent_state: dict[str, float], t: int | None = None) -> dict:
        if t is None:
            raise ValueError("WindblownSensor.observe requires time index t")
        self._ensure_legacy_direct_observe_ready()
        if not self.can_sample(t):
            status = self.get_status()
            return {
                "sensor_id": self.sensor_id,
                "available": False,
                "variables": list(self.spec.variables),
                "power_cost": self.power_cost(),
                "mode": status["mode"],
                "warm_remaining_steps": status["warm_remaining_steps"],
            }
        values = np.asarray([float(latent_state[v]) for v in self.spec.variables], dtype=float)
        values = values + self._noise_vec()
        self._touch_sampled(t)
        status = self.get_status()
        return {
            "sensor_id": self.sensor_id,
            "available": True,
            "y": values,
            "variables": list(self.spec.variables),
            "power_cost": self.power_cost(),
            "mode": status["mode"],
            "warm_remaining_steps": status["warm_remaining_steps"],
        }
