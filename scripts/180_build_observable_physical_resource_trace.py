#!/usr/bin/env python3
"""Build a deployment-observable heater/resource trace.

The controller uses only weather variables available before scheduling.  The
generator mode, subtype labels, and future targets are deliberately ignored.
Physical heater increments come from the declared hardware manifest; the
normalized costs retain the fixed acquisition-frequency factors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = {
    "met_station_core": (0.72, 0.30, 4.50),
    "radiometer_basic": (0.68, 0.36, 0.00),
    "surface_temp_ir": (0.76, 0.0156, 0.00),
    "laser_disdrometer": (0.88, 1.50, 50.00),
    "fc4_flux": (0.82, 0.06, 0.00),
}
BACKBONE = 0.4104
PHYSICAL_BUDGET_W = 55.0
EFFECTIVE_BUDGET = 2.15


def _dew_point_celsius(temperature: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    # Magnus approximation, adequate for the controller screen and explicit in
    # the output manifest.  RH is clipped to avoid invalid logarithms.
    rh = np.clip(relative_humidity, 1.0, 100.0)
    gamma = np.log(rh / 100.0) + 17.625 * temperature / (243.04 + temperature)
    return 243.04 * gamma / (17.625 - gamma)


def _hysteresis(on_condition: np.ndarray, off_condition: np.ndarray) -> np.ndarray:
    state = np.zeros(len(on_condition), dtype=np.int8)
    for i in range(1, len(state)):
        if state[i - 1]:
            state[i] = 0 if bool(off_condition[i]) else 1
        else:
            state[i] = 1 if bool(on_condition[i]) else 0
    return state


def build(frame: pd.DataFrame, *, use_nowcast: bool = False) -> tuple[pd.DataFrame, dict]:
    required = {
        "time_idx",
        "air_temperature_c",
        "relative_humidity",
        "snow_surface_temperature_c",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing controller inputs: {missing}")

    out = frame[["time_idx"]].copy()
    controller_frame = frame
    if use_nowcast:
        required_nowcast = {
            "agent_context_nowcast_air_temperature_c": "air_temperature_c",
            "agent_context_nowcast_relative_humidity": "relative_humidity",
        }
        if not set(required_nowcast).issubset(frame.columns):
            raise ValueError("nowcast resource control requires air temperature and humidity nowcasts")
        controller_frame = frame.copy()
        controller_frame["air_temperature_c"] = frame["agent_context_nowcast_air_temperature_c"]
        controller_frame["relative_humidity"] = frame["agent_context_nowcast_relative_humidity"]
        controller_frame["snow_surface_temperature_c"] = frame["snow_surface_temperature_c"]
    temperature = controller_frame["air_temperature_c"].to_numpy(float)
    humidity = controller_frame["relative_humidity"].to_numpy(float)
    surface = controller_frame["snow_surface_temperature_c"].to_numpy(float)
    dew_point = _dew_point_celsius(temperature, humidity)
    margin = temperature - dew_point
    # These thresholds are copied from the declared physical manifest.  The
    # controller uses current weather only and retains state through hysteresis.
    gmx_on = (temperature <= -25.0) | ((margin <= 1.0) & (temperature <= -5.0))
    gmx_off = (temperature >= -23.0) & (margin >= 2.0)
    parsivel_on = (temperature <= -20.0) | ((margin <= 1.0) & (surface <= -5.0))
    parsivel_off = (temperature >= -18.0) & (margin >= 2.0)
    states = {
        "met_station_core": _hysteresis(gmx_on, gmx_off),
        "laser_disdrometer": _hysteresis(parsivel_on, parsivel_off),
    }
    for channel in CHANNELS:
        state = states.get(channel, np.zeros(len(frame), dtype=np.int8))
        base_effective, base_physical, increment_physical = CHANNELS[channel]
        increment_effective = increment_physical * EFFECTIVE_BUDGET / PHYSICAL_BUDGET_W
        out[f"resource_heater_on_{channel}"] = state
        out[f"resource_power_w_{channel}"] = base_physical + state * increment_physical
        out[f"resource_effective_power_{channel}"] = base_effective + state * increment_effective
    effective_cols = [f"resource_effective_power_{c}" for c in CHANNELS]
    physical_cols = [f"resource_power_w_{c}" for c in CHANNELS]
    out["resource_dew_point_c"] = dew_point
    out["resource_dew_point_margin_c"] = margin
    out["resource_optional_power_w"] = out[physical_cols].sum(axis=1)
    out["resource_total_power_w"] = out["resource_optional_power_w"] + 0.4104
    out["resource_optional_effective_power"] = out[effective_cols].sum(axis=1)
    out["resource_total_effective_power"] = out["resource_optional_effective_power"] + BACKBONE
    out["resource_absolute_feasible_all_optional"] = out["resource_total_power_w"] <= PHYSICAL_BUDGET_W
    out["resource_effective_feasible_all_optional"] = out["resource_total_effective_power"] <= EFFECTIVE_BUDGET
    metadata = {
        "generator": Path(__file__).name,
        "controller_inputs": ["air_temperature_c", "relative_humidity", "snow_surface_temperature_c"],
        "uses_exact_generator_mode": False,
        "uses_future_targets": False,
        "uses_event_labels": False,
        "hysteresis": {
            "gmx500_on": "air_temperature <= -25 C or (dew_point_margin <= 1 C and air_temperature <= -5 C)",
            "gmx500_off": "air_temperature >= -23 C and dew_point_margin >= 2 C",
            "parsivel_on": "air_temperature <= -20 C or (dew_point_margin <= 1 C and surface_temperature <= -5 C)",
            "parsivel_off": "air_temperature >= -18 C and dew_point_margin >= 2 C",
        },
        "physical_budget_w": PHYSICAL_BUDGET_W,
        "effective_budget": EFFECTIVE_BUDGET,
        "fixed_frequency_cost_is_action": False,
        "uses_decision_time_nowcast": bool(use_nowcast),
        "channels": CHANNELS,
    }
    return out, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-nowcast", action="store_true")
    args = parser.parse_args()
    result, metadata = build(pd.read_csv(args.truth), use_nowcast=bool(args.use_nowcast))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(result),
        "heater_duty": {c: float(result[f"resource_heater_on_{c}"].mean()) for c in CHANNELS},
        "effective_power_states": int(result["resource_total_effective_power"].nunique()),
        "all_optional_feasible_fraction": float(result["resource_effective_feasible_all_optional"].mean()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
