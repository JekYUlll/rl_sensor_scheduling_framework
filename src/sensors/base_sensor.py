from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.interfaces import BaseSensor


@dataclass
class SensorSpec:
    sensor_id: str
    obs_dim: int
    variables: list[str]
    refresh_interval: int
    power_cost: float
    startup_delay: int = 0
    noise_std: float | list[float] = 0.0


class AbstractSensor(BaseSensor):
    def __init__(self, spec: SensorSpec) -> None:
        self.spec = spec
        self._last_sample_time = -10**9

    @property
    def sensor_id(self) -> str:
        return self.spec.sensor_id

    def power_cost(self) -> float:
        return float(self.spec.power_cost)

    def can_sample(self, t: int) -> bool:
        if t < self.spec.startup_delay:
            return False
        return (t - self._last_sample_time) >= self.spec.refresh_interval

    def _touch_sampled(self, t: int) -> None:
        self._last_sample_time = t

    def _noise_vec(self) -> np.ndarray:
        if isinstance(self.spec.noise_std, list):
            std = np.asarray(self.spec.noise_std, dtype=float)
        else:
            std = np.full(self.spec.obs_dim, float(self.spec.noise_std), dtype=float)
        return np.random.randn(self.spec.obs_dim) * std

    def observe(self, latent_state: Any) -> dict:
        raise NotImplementedError
