from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SensorSpec:
    sensor_id: str
    obs_dim: int
    variables: list[str]
    refresh_interval: int
    power_cost: float
    startup_delay: int = 0
    noise_std: float | list[float] = 0.0
