"""Entity-level power accounting for the six-component AWS system.

The values are transcribed from local equipment notes and manufacturer
documentation.  This module is deliberately separate from the normalized
research scene costs: an experiment may use it only after channel mapping and
the supply-side limit have been declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DevicePower:
    device_id: str
    nominal_voltage_v: float
    steady_power_w: float
    active_power_w: float
    peak_power_w: float
    mandatory: bool = False
    source: str = ""


# The power values are conservative operating points, not measured system
# telemetry.  Parsivel2's heater is a transient peak and is not its baseline
# consumption.
ENTITY_POWER = {
    "gmx500": DevicePower(
        "gmx500", 12.0, 0.30, 0.30, 0.30,
        source="Documents/Academic/传感器-2025-10-10.pptx",
    ),
    "fc4": DevicePower(
        "fc4", 12.0, 0.060, 0.060, 0.240,
        source="Documents/Academic/传感器-2025-10-10.pptx",
    ),
    "parsivel2": DevicePower(
        "parsivel2", 24.0, 1.50, 1.50, 100.0,
        source="Documents/Academic/Parsivel2_en.pdf;传感器-2025-10-10.pptx",
    ),
    "lps10": DevicePower(
        "lps10", 24.0, 0.360, 0.360, 0.360,
        source="Senseca LPS10 datasheet, DSH_LPS10_EN_datasheet.pdf",
    ),
    "si111": DevicePower(
        "si111", 12.0, 0.0156, 0.0156, 0.0156,
        source="Documents/Academic/传感器-2025-10-10.pptx",
    ),
    "cr1000xe": DevicePower(
        "cr1000xe", 24.0, 0.4104, 0.4104, 0.4104, mandatory=True,
        source="Campbell Scientific CR1000Xe power requirements",
    ),
}

# Only these five channels have a documented sensing-device mapping.  The
# logger is a mandatory backbone, not an action selectable by the scheduler.
CHANNEL_DEVICE = {
    "met_station_core": "gmx500",
    "radiometer_basic": "lps10",
    "surface_temp_ir": "si111",
    "laser_disdrometer": "parsivel2",
    "fc4_flux": "fc4",
}


def validate_channel_mapping(channel_ids: list[str] | tuple[str, ...]) -> None:
    unknown = sorted(set(channel_ids) - set(CHANNEL_DEVICE))
    if unknown:
        raise ValueError(
            "Unmapped physical channels: " + ", ".join(unknown)
            + ". Add a documented device mapping before physical experiments."
        )


def system_power_w(
    selected_channels: list[str] | tuple[str, ...],
    *,
    heater_on: bool = False,
    active_channels: set[str] | None = None,
) -> dict[str, float]:
    """Return backbone, steady, peak and total load in watts.

    `selected_channels` contains measurement channels.  CR1000Xe is always
    included.  `active_channels` selects the documented higher acquisition
    state where a device has one; it currently leaves the conservative values
    unchanged and is retained for future measured duty-cycle calibration.
    """
    validate_channel_mapping(selected_channels)
    active = set(active_channels or ())
    selected_devices = {CHANNEL_DEVICE[c] for c in selected_channels}
    selected_devices.add("cr1000xe")
    steady = 0.0
    active_power = 0.0
    peak = 0.0
    for device_id in selected_devices:
        device = ENTITY_POWER[device_id]
        steady += device.steady_power_w
        active_power += device.active_power_w if device_id in active else device.steady_power_w
        if device_id == "parsivel2" and not heater_on:
            peak += device.steady_power_w
        else:
            peak += device.peak_power_w
    return {
        "backbone_power_w": ENTITY_POWER["cr1000xe"].steady_power_w,
        "steady_power_w": steady,
        "active_power_w": active_power,
        "peak_power_w": peak,
    }


def ledger_rows() -> list[Mapping[str, object]]:
    return [
        {
            "channel_id": channel,
            "device_id": device,
            "device_voltage_v": ENTITY_POWER[device].nominal_voltage_v,
            "steady_power_w": ENTITY_POWER[device].steady_power_w,
            "active_power_w": ENTITY_POWER[device].active_power_w,
            "peak_power_w": ENTITY_POWER[device].peak_power_w,
            "mandatory": False,
            "source": ENTITY_POWER[device].source,
        }
        for channel, device in CHANNEL_DEVICE.items()
    ] + [{
        "channel_id": "logger_backbone",
        "device_id": "cr1000xe",
        "device_voltage_v": ENTITY_POWER["cr1000xe"].nominal_voltage_v,
        "steady_power_w": ENTITY_POWER["cr1000xe"].steady_power_w,
        "active_power_w": ENTITY_POWER["cr1000xe"].active_power_w,
        "peak_power_w": ENTITY_POWER["cr1000xe"].peak_power_w,
        "mandatory": True,
        "source": ENTITY_POWER["cr1000xe"].source,
    }]
