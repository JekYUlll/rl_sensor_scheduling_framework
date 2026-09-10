#!/usr/bin/env python3
"""Add the physical radiometer heater/ventilator load to the V681 trace.

The V681 core and laser rules are unchanged.  The radiometer receives a
15-W auxiliary load representing the declared heater plus ventilator operating
point, driven by causal cold, thermal-mode proxy, and low-radiation inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = {
    "met_station_core": (0.72, 0.30, 4.50),
    "radiometer_basic": (0.68, 0.36, 15.00),
    "surface_temp_ir": (0.76, 0.0156, 0.00),
    "laser_disdrometer": (0.88, 1.50, 50.00),
    "fc4_flux": (0.82, 0.06, 0.00),
}
BACKBONE = 0.4104
EFFECTIVE_BUDGET = 2.50
PHYSICAL_BUDGET = 55.0
ON_THRESHOLD = 0.50
OFF_THRESHOLD = 0.35


def _hysteresis(demand: np.ndarray) -> np.ndarray:
    state = np.zeros(len(demand), dtype=np.int8)
    for index in range(1, len(state)):
        state[index] = 0 if state[index - 1] and demand[index] <= OFF_THRESHOLD else int(demand[index] >= ON_THRESHOLD)
    return state


def build(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "time_idx",
        "agent_context_forecast_mode_transport",
        "agent_context_forecast_mode_particle",
        "agent_context_forecast_mode_thermal",
        "generator_observable_wind_driver",
        "generator_observable_thermal_driver",
        "agent_context_nowcast_solar_radiation_wm2",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"truth is missing V683 inputs: {missing}")
    out = frame[["time_idx"]].copy()
    mode = {name: np.clip(frame[f"agent_context_forecast_mode_{name}"].to_numpy(float), 0.0, 1.0) for name in ("transport", "particle", "thermal")}
    wind = np.clip((frame["generator_observable_wind_driver"].to_numpy(float) + 1.0) / 2.0, 0.0, 1.0)
    cold = np.clip((-frame["generator_observable_thermal_driver"].to_numpy(float) + 1.0) / 2.0, 0.0, 1.0)
    solar = np.clip(frame["agent_context_nowcast_solar_radiation_wm2"].to_numpy(float) / 250.0, 0.0, 1.0)
    low_radiation = 1.0 - solar
    demands = {
        "met_station_core": np.clip(0.25 + 0.45 * mode["thermal"] + 0.15 * mode["transport"] + 0.15 * cold, 0.0, 1.0),
        "laser_disdrometer": np.clip(0.20 + 0.40 * mode["particle"] + 0.25 * mode["transport"] + 0.15 * wind, 0.0, 1.0),
        # The coefficients represent a moderate frost-protection duty, not a
        # permanent cold-start trigger; they were fixed before final-window
        # inspection and keep the heater state informative but intermittent.
        "radiometer_basic": np.clip(0.12 + 0.30 * cold + 0.18 * mode["thermal"] + 0.10 * low_radiation, 0.0, 1.0),
    }
    states = {channel: _hysteresis(demand) for channel, demand in demands.items()}
    for channel, demand in demands.items():
        out[f"resource_heater_demand_{channel}"] = demand
    for channel, (base_effective, base_physical, increment_physical) in CHANNELS.items():
        state = states.get(channel, np.zeros(len(frame), dtype=np.int8))
        out[f"resource_heater_on_{channel}"] = state
        out[f"resource_power_w_{channel}"] = base_physical + state * increment_physical
        out[f"resource_effective_power_{channel}"] = base_effective + state * increment_physical * EFFECTIVE_BUDGET / PHYSICAL_BUDGET
    effective = [f"resource_effective_power_{channel}" for channel in CHANNELS]
    physical = [f"resource_power_w_{channel}" for channel in CHANNELS]
    out["resource_optional_effective_power"] = out[effective].sum(axis=1)
    out["resource_total_effective_power"] = out["resource_optional_effective_power"] + BACKBONE
    out["resource_optional_power_w"] = out[physical].sum(axis=1)
    out["resource_total_power_w"] = out["resource_optional_power_w"] + BACKBONE
    out["resource_effective_feasible_all_optional"] = out["resource_total_effective_power"] <= EFFECTIVE_BUDGET
    return out, {
        "generator": Path(__file__).name,
        "parent_controller": "v681_v527_observable_mode_proxy_wind_cold_hysteresis",
        "radiometer_auxiliary_load_w": 15.0,
        "radiometer_load_basis": "heater plus ventilator operating point",
        "policy_inputs": ["forecast_mode_proxies", "observable_wind_thermal_drivers", "nowcast_solar_radiation"],
        "thresholds": {"on": ON_THRESHOLD, "off": OFF_THRESHOLD},
        "effective_budget": EFFECTIVE_BUDGET,
        "physical_budget_reference_w": PHYSICAL_BUDGET,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, metadata = build(pd.read_csv(args.truth))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(result),
        "heater_duty": {channel: float(result[f"resource_heater_on_{channel}"].mean()) for channel in CHANNELS},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
