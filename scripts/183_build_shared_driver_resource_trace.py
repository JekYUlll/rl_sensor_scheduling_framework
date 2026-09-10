#!/usr/bin/env python3
"""Derive conditional effective loads from the shared physical drivers.

The drivers are truth-side projections of decision-time weather nowcasts.  The
controller models increased anti-icing/particle-load demand for the exposed
GMX500 and Parsivel channels, with hysteresis.  Fixed acquisition-frequency
costs remain fixed and are not actions.
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
EFFECTIVE_BUDGET = 2.15
PHYSICAL_BUDGET = 55.0


def hysteresis(on: np.ndarray, off: np.ndarray) -> np.ndarray:
    state = np.zeros(len(on), dtype=np.int8)
    for i in range(1, len(state)):
        state[i] = 0 if (state[i - 1] and off[i]) else int(on[i])
    return state


def build(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    names = ("transport", "particle", "thermal")
    required = {"time_idx", *(f"generator_shared_{name}_driver" for name in names)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"truth is missing shared drivers: {missing}")
    out = frame[["time_idx"]].copy()
    d = {name: np.clip(frame[f"generator_shared_{name}_driver"].to_numpy(float), 0.0, 1.0) for name in names}
    core_demand = np.clip(0.25 + 0.55 * d["thermal"] + 0.20 * d["transport"], 0.0, 1.0)
    laser_demand = np.clip(0.20 + 0.55 * d["particle"] + 0.25 * d["transport"], 0.0, 1.0)
    states = {
        "met_station_core": hysteresis(core_demand >= 0.50, core_demand <= 0.35),
        "laser_disdrometer": hysteresis(laser_demand >= 0.50, laser_demand <= 0.35),
    }
    out["resource_shared_exposure_load"] = np.maximum.reduce(tuple(d.values()))
    out["resource_heater_demand_met_station_core"] = core_demand
    out["resource_heater_demand_laser_disdrometer"] = laser_demand
    for channel, (base_effective, base_physical, increment_physical) in CHANNELS.items():
        state = states.get(channel, np.zeros(len(frame), dtype=np.int8))
        out[f"resource_heater_on_{channel}"] = state
        out[f"resource_power_w_{channel}"] = base_physical + state * increment_physical
        out[f"resource_effective_power_{channel}"] = base_effective + state * increment_physical * EFFECTIVE_BUDGET / PHYSICAL_BUDGET
    effective = [f"resource_effective_power_{c}" for c in CHANNELS]
    physical = [f"resource_power_w_{c}" for c in CHANNELS]
    out["resource_optional_effective_power"] = out[effective].sum(axis=1)
    out["resource_total_effective_power"] = out["resource_optional_effective_power"] + BACKBONE
    out["resource_optional_power_w"] = out[physical].sum(axis=1)
    out["resource_total_power_w"] = out["resource_optional_power_w"] + BACKBONE
    out["resource_effective_feasible_all_optional"] = out["resource_total_effective_power"] <= EFFECTIVE_BUDGET
    return out, {
        "generator": Path(__file__).name,
        "controller": "shared_transport_particle_thermal_driver_hysteresis",
        "fixed_frequency_cost_is_action": False,
        "effective_budget": EFFECTIVE_BUDGET,
        "physical_budget_reference_w": PHYSICAL_BUDGET,
        "thresholds": {"core_on": 0.50, "core_off": 0.35, "laser_on": 0.50, "laser_off": 0.35},
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
    print(json.dumps({"rows": len(result), "heater_duty": {c: float(result[f"resource_heater_on_{c}"].mean()) for c in CHANNELS}}, sort_keys=True))


if __name__ == "__main__":
    main()
