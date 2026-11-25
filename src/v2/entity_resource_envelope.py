"""Documented physical resource-envelope constants and subset enumeration."""

from __future__ import annotations

import itertools

from v2.entity_energy import CHANNEL_DEVICE, system_power_w

BATTERY_CAPACITY_WH = 8640.0
CONTROLLER_VOLTAGE_V = 24.0
CONTROLLER_CURRENT_A = 50.0
CONTROLLER_POWER_W = CONTROLLER_VOLTAGE_V * CONTROLLER_CURRENT_A
PV_RATED_W = 600.0
WIND_RATED_W = 400.0
# Documented continuous auxiliary loads from the system-design sheet:
# STM32 <=1.3 W, RK3568 <=2.84 W, and cold-rated camera 1.87 W.
FIXED_AUXILIARY_POWER_W = 1.30 + 2.84 + 1.87


def rows() -> list[dict[str, object]]:
    channels = tuple(CHANNEL_DEVICE)
    result: list[dict[str, object]] = []
    for size in range(len(channels) + 1):
        for subset in itertools.combinations(channels, size):
            normal = system_power_w(list(subset), heater_on=False)
            heater = system_power_w(list(subset), heater_on=True)
            steady = float(normal["steady_power_w"])
            peak = float(heater["peak_power_w"])
            system_steady = steady + FIXED_AUXILIARY_POWER_W
            system_peak = peak + FIXED_AUXILIARY_POWER_W
            result.append(
                {
                    "selected_channels": ",".join(subset),
                    "selected_count": size,
                    "steady_power_w": steady,
                    "heater_peak_power_w": peak,
                    "system_steady_power_w": system_steady,
                    "system_heater_peak_power_w": system_peak,
                    "daily_steady_energy_wh": steady * 24.0,
                    "system_daily_steady_energy_wh": system_steady * 24.0,
                    "battery_hours_at_steady_load": BATTERY_CAPACITY_WH / system_steady,
                    "controller_steady_feasible": system_steady <= CONTROLLER_POWER_W,
                    "controller_heater_peak_feasible": system_peak <= CONTROLLER_POWER_W,
                }
            )
    return result
