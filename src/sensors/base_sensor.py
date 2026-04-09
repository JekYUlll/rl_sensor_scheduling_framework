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
    warmup_steps: int = 0
    startup_peak_power: float | None = None
    required: bool = False
    warmup_observation_mode: str = "none"
    warmup_noise_scale: float = 1.0
    noise_std: float | list[float] | dict[str, float] = 0.0


class AbstractSensor(BaseSensor):
    def __init__(self, spec: SensorSpec) -> None:
        self.spec = spec
        self.reset()

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
        if not self.is_ready():
            return False
        return (t - self._last_sample_time) >= self.spec.refresh_interval

    def _touch_sampled(self, t: int) -> None:
        self._last_sample_time = t

    def reset(self) -> None:
        self._last_sample_time = -10**9
        self._powered = False
        self._mode = "off"
        self._time_since_power_on: int | None = None
        self._warm_remaining_steps = 0

    def warmup_steps(self) -> int:
        return max(int(self.spec.warmup_steps), 0)

    def is_powered(self) -> bool:
        return bool(self._powered)

    def is_warming(self) -> bool:
        return str(self._mode) == "warming"

    def is_ready(self) -> bool:
        return str(self._mode) == "ready"

    def warm_remaining_steps(self) -> int:
        return int(self._warm_remaining_steps)

    def time_since_power_on(self) -> int | None:
        return None if self._time_since_power_on is None else int(self._time_since_power_on)

    def _update_mode(self, t: int) -> None:
        if not self._powered:
            self._mode = "off"
            self._warm_remaining_steps = 0
            return
        _ = int(t)
        elapsed = int(self._time_since_power_on or 0)
        warm_remaining = max(self.warmup_steps() - elapsed, 0)
        startup_remaining = max(int(self.spec.startup_delay) - elapsed, 0)
        self._warm_remaining_steps = int(max(warm_remaining, startup_remaining))
        self._mode = "warming" if self._warm_remaining_steps > 0 else "ready"

    def _ensure_legacy_direct_observe_ready(self) -> None:
        # Preserve the old direct-observe behavior for sensors that have no
        # warm-up semantics at all. New warm-up-enabled sensors must still go
        # through begin_step() so the environment can charge power correctly.
        if self._powered:
            return
        if self.warmup_steps() > 0 or int(self.spec.startup_delay) > 0:
            return
        self._powered = True
        self._mode = "ready"
        self._time_since_power_on = 0
        self._warm_remaining_steps = 0

    def begin_step(self, selected: bool, t: int) -> dict[str, float | int | bool | str]:
        if not bool(selected):
            self._powered = False
            self._mode = "off"
            self._time_since_power_on = None
            self._warm_remaining_steps = 0
            return self.get_status()
        if not self._powered:
            self._powered = True
            self._time_since_power_on = 0
        elif self._time_since_power_on is None:
            self._time_since_power_on = 0
        else:
            self._time_since_power_on += 1
        self._update_mode(int(t))
        return self.get_status()

    def get_status(self) -> dict[str, float | int | bool | str]:
        return {
            "sensor_id": self.sensor_id,
            "mode": str(self._mode),
            "powered": bool(self._powered),
            "warming": bool(self.is_warming()),
            "ready": bool(self.is_ready()),
            "warm_remaining_steps": int(self._warm_remaining_steps),
            "time_since_power_on": -1 if self._time_since_power_on is None else int(self._time_since_power_on),
            "power_cost": float(self.power_cost()),
        }

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
