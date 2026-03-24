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
    startup_peak_power: float | None = None
    required: bool = False
    noise_std: float | list[float] | dict[str, float] = 0.0


class AbstractSensor(BaseSensor):
    def __init__(self, spec: SensorSpec) -> None:
        self.spec = spec
        self._last_sample_time = -10**9

    @property
    def sensor_id(self) -> str:
        return self.spec.sensor_id

    def power_cost(self) -> float:
        return float(self.spec.power_cost)

    def startup_peak_power(self) -> float:
        if self.spec.startup_peak_power is None:
            return float(self.spec.power_cost)
        return float(self.spec.startup_peak_power)

    def can_sample(self, t: int) -> bool:
        if t < self.spec.startup_delay:
            return False
        return (t - self._last_sample_time) >= self.spec.refresh_interval

    def _touch_sampled(self, t: int) -> None:
        self._last_sample_time = t

    def reset(self) -> None:
        self._last_sample_time = -10**9

    def _noise_std_vec(self) -> np.ndarray:
        noise_std = self.spec.noise_std
        if isinstance(noise_std, dict):
            std = np.asarray([float(noise_std.get(name, 0.0)) for name in self.spec.variables], dtype=float)
        elif isinstance(noise_std, list):
            std = np.asarray(noise_std, dtype=float)
        else:
            std = np.full(self.spec.obs_dim, float(noise_std), dtype=float)
        if std.shape[0] != self.spec.obs_dim:
            raise ValueError(
                f"Sensor {self.sensor_id} expected noise vector of length {self.spec.obs_dim}, got {std.shape[0]}"
            )
        return std

    def _noise_vec(self) -> np.ndarray:
        std = self._noise_std_vec()
        return np.random.randn(self.spec.obs_dim) * std

    def observe(self, latent_state: Any) -> dict:
        raise NotImplementedError
