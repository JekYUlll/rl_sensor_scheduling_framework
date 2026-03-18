from __future__ import annotations

import numpy as np

from sensors.base_sensor import AbstractSensor, SensorSpec


class DatasetSensor(AbstractSensor):
    def __init__(self, spec: SensorSpec, state_columns: list[str]) -> None:
        super().__init__(spec)
        self.state_columns = list(state_columns)
        self.var_to_idx = {name: i for i, name in enumerate(self.state_columns)}
        self.obs_indices = [self.var_to_idx[v] for v in self.spec.variables]

    def observe(self, latent_state: dict[str, float], t: int | None = None) -> dict:
        if t is None:
            raise ValueError("DatasetSensor.observe requires time index t")
        if not self.can_sample(t):
            return {"sensor_id": self.sensor_id, "available": False}

        values = np.asarray([float(latent_state[v]) for v in self.spec.variables], dtype=float)
        noise = self._noise_vec()
        y = values + noise

        c_mat = np.zeros((len(self.obs_indices), len(self.state_columns)), dtype=float)
        for row_idx, col_idx in enumerate(self.obs_indices):
            c_mat[row_idx, col_idx] = 1.0

        if isinstance(self.spec.noise_std, list):
            std = np.asarray(self.spec.noise_std, dtype=float)
        else:
            std = np.full(len(self.obs_indices), float(self.spec.noise_std), dtype=float)
        r_mat = np.diag(np.maximum(std, 1e-6) ** 2)

        self._touch_sampled(t)
        return {
            "sensor_id": self.sensor_id,
            "available": True,
            "y": y,
            "C": c_mat,
            "R": r_mat,
            "variables": list(self.spec.variables),
            "power_cost": self.power_cost(),
        }
