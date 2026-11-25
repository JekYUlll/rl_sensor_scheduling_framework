from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SensorSpecV2:
    sensor_id: str
    observed_variables: tuple[str, ...]
    power_cost: float
    startup_peak_power: float
    warmup_steps: int = 0
    sampling_interval: int = 1
    noise_std: dict[str, float] = field(default_factory=dict)
    calm_noise_std: dict[str, float] = field(default_factory=dict)
    calm_noise_multiplier: dict[str, float] = field(default_factory=dict)
    calm_observation_probability: dict[str, float] = field(default_factory=dict)
    event_noise_std: dict[str, float] = field(default_factory=dict)
    event_noise_multiplier: dict[str, float] = field(default_factory=dict)
    event_observation_probability: dict[str, float] = field(default_factory=dict)
    event_subtype_noise_std: dict[int, dict[str, float]] = field(default_factory=dict)
    event_subtype_observation_probability: dict[int, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "SensorSpecV2":
        variables = item.get("observed_variables", item.get("variables", []))
        peak = item.get("startup_peak_power", item.get("power_cost", 0.0))
        return cls(
            sensor_id=str(item["sensor_id"]),
            observed_variables=tuple(str(v) for v in variables),
            power_cost=float(item.get("power_cost", 0.0)),
            startup_peak_power=float(peak),
            warmup_steps=int(item.get("warmup_steps", 0)),
            sampling_interval=int(item.get("sampling_interval", item.get("refresh_interval", 1))),
            noise_std={str(k): float(v) for k, v in dict(item.get("noise_std", {}) or {}).items()},
            calm_noise_std={str(k): float(v) for k, v in dict(item.get("calm_noise_std", {}) or {}).items()},
            calm_noise_multiplier={
                str(k): float(v) for k, v in dict(item.get("calm_noise_multiplier", {}) or {}).items()
            },
            calm_observation_probability={
                str(k): float(v) for k, v in dict(item.get("calm_observation_probability", {}) or {}).items()
            },
            event_noise_std={str(k): float(v) for k, v in dict(item.get("event_noise_std", {}) or {}).items()},
            event_noise_multiplier={
                str(k): float(v) for k, v in dict(item.get("event_noise_multiplier", {}) or {}).items()
            },
            event_observation_probability={
                str(k): float(v) for k, v in dict(item.get("event_observation_probability", {}) or {}).items()
            },
            event_subtype_noise_std=_parse_subtype_variable_map(item.get("event_subtype_noise_std", {}) or {}),
            event_subtype_observation_probability=_parse_subtype_variable_map(
                item.get("event_subtype_observation_probability", {}) or {}
            ),
        )


def load_sensor_specs(path: str | Path) -> list[SensorSpecV2]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [SensorSpecV2.from_mapping(item) for item in data.get("sensors", [])]


def _subtype_key(value: object) -> int:
    names = {
        "calm": 0,
        "none": 0,
        "particle": 1,
        "flux": 2,
        "thermal": 3,
        "thermal_boundary": 3,
    }
    text = str(value).strip().lower()
    if text in names:
        return int(names[text])
    return int(text)


def _parse_subtype_variable_map(raw: object) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for subtype, values in dict(raw or {}).items():
        out[_subtype_key(subtype)] = {
            str(name): float(value)
            for name, value in dict(values or {}).items()
        }
    return out
