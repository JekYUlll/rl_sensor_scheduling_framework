"""Deterministic external-load profiles for the entity energy model."""

from __future__ import annotations

import numpy as np


def build_hysteretic_heater_profile(
    surface_temperature_c: np.ndarray,
    *,
    on_threshold_c: float = -25.0,
    off_threshold_c: float = -23.0,
    heater_power_w: float = 600.0,
) -> np.ndarray:
    """Return a non-controllable heater load with temperature hysteresis.

    The heater is an external fixed load. It turns on at or below the lower
    threshold and remains on until the temperature reaches the upper threshold.
    No policy output is used to construct this profile.
    """
    temperatures = np.asarray(surface_temperature_c, dtype=float).reshape(-1)
    if temperatures.size == 0:
        return np.zeros(0, dtype=float)
    if not np.all(np.isfinite(temperatures)):
        raise ValueError("surface temperatures must be finite")
    if not np.isfinite(on_threshold_c) or not np.isfinite(off_threshold_c):
        raise ValueError("heater thresholds must be finite")
    if off_threshold_c <= on_threshold_c:
        raise ValueError("off_threshold_c must be greater than on_threshold_c")
    if not np.isfinite(heater_power_w) or heater_power_w < 0.0:
        raise ValueError("heater_power_w must be finite and non-negative")

    on = False
    profile = np.zeros(len(temperatures), dtype=float)
    for idx, temperature in enumerate(temperatures):
        if not on and temperature <= on_threshold_c:
            on = True
        elif on and temperature >= off_threshold_c:
            on = False
        profile[idx] = heater_power_w if on else 0.0
    return profile
