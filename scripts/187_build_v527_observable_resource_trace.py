#!/usr/bin/env python3
"""Build a causal effective-resource trace matched to V527-r2 truth.

The controller uses only columns that are available to the scheduler at run
time: noisy forecast-mode scores and the causal wind/thermal nowcast drivers.
It deliberately does not read persistent mode ids, target innovations, or
future observations.  The power constants and hysteresis thresholds are
frozen before downstream asset fitting.
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
EFFECTIVE_BUDGET = 2.50
PHYSICAL_BUDGET = 55.0
ON_THRESHOLD = 0.50
OFF_THRESHOLD = 0.35


def _hysteresis(demand: np.ndarray) -> np.ndarray:
    state = np.zeros(len(demand), dtype=np.int8)
    for index in range(1, len(state)):
        state[index] = (
            0
            if state[index - 1] and demand[index] <= OFF_THRESHOLD
            else int(demand[index] >= ON_THRESHOLD)
        )
    return state


def build(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "time_idx",
        "agent_context_forecast_mode_transport",
        "agent_context_forecast_mode_particle",
        "agent_context_forecast_mode_thermal",
        "generator_observable_wind_driver",
        "generator_observable_thermal_driver",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"truth is missing observable resource inputs: {missing}")

    out = frame[["time_idx"]].copy()
    mode = {
        name: np.clip(
            frame[f"agent_context_forecast_mode_{name}"].to_numpy(dtype=float), 0.0, 1.0
        )
        for name in ("transport", "particle", "thermal")
    }
    wind = np.clip(
        (frame["generator_observable_wind_driver"].to_numpy(dtype=float) + 1.0) / 2.0,
        0.0,
        1.0,
    )
    cold = np.clip(
        (-frame["generator_observable_thermal_driver"].to_numpy(dtype=float) + 1.0) / 2.0,
        0.0,
        1.0,
    )
    # Demand is a deployable proxy: mode scores carry the event context, while
    # wind and coldness provide continuous physical loading.  Exact generator
    # mode and target innovation are intentionally excluded.
    core_demand = np.clip(
        0.25 + 0.45 * mode["thermal"] + 0.15 * mode["transport"] + 0.15 * cold,
        0.0,
        1.0,
    )
    laser_demand = np.clip(
        0.20 + 0.40 * mode["particle"] + 0.25 * mode["transport"] + 0.15 * wind,
        0.0,
        1.0,
    )
    heater_state = {
        "met_station_core": _hysteresis(core_demand),
        "laser_disdrometer": _hysteresis(laser_demand),
    }
    out["resource_observable_wind_load"] = wind
    out["resource_observable_cold_load"] = cold
    out["resource_heater_demand_met_station_core"] = core_demand
    out["resource_heater_demand_laser_disdrometer"] = laser_demand
    for channel, (base_effective, base_physical, increment_physical) in CHANNELS.items():
        state = heater_state.get(channel, np.zeros(len(frame), dtype=np.int8))
        out[f"resource_heater_on_{channel}"] = state
        out[f"resource_power_w_{channel}"] = base_physical + state * increment_physical
        out[f"resource_effective_power_{channel}"] = (
            base_effective + state * increment_physical * EFFECTIVE_BUDGET / PHYSICAL_BUDGET
        )
    effective = [f"resource_effective_power_{channel}" for channel in CHANNELS]
    physical = [f"resource_power_w_{channel}" for channel in CHANNELS]
    out["resource_optional_effective_power"] = out[effective].sum(axis=1)
    out["resource_total_effective_power"] = out["resource_optional_effective_power"] + BACKBONE
    out["resource_optional_power_w"] = out[physical].sum(axis=1)
    out["resource_total_power_w"] = out["resource_optional_power_w"] + BACKBONE
    out["resource_effective_feasible_all_optional"] = (
        out["resource_total_effective_power"] <= EFFECTIVE_BUDGET
    )
    metadata = {
        "generator": Path(__file__).name,
        "controller": "v527_observable_mode_proxy_wind_cold_hysteresis",
        "policy_inputs": [
            "agent_context_forecast_mode_transport",
            "agent_context_forecast_mode_particle",
            "agent_context_forecast_mode_thermal",
            "generator_observable_wind_driver",
            "generator_observable_thermal_driver",
        ],
        "excluded_generator_state": [
            "generator_persistent_mode_id",
            "generator_independent_target_innovation",
        ],
        "effective_budget": EFFECTIVE_BUDGET,
        "physical_budget_reference_w": PHYSICAL_BUDGET,
        "fixed_frequency_cost_is_action": False,
        "thresholds": {"on": ON_THRESHOLD, "off": OFF_THRESHOLD},
        "demand_formula": {
            "core": "0.25 + 0.45*thermal_score + 0.15*transport_score + 0.15*cold_load",
            "laser": "0.20 + 0.40*particle_score + 0.25*transport_score + 0.15*wind_load",
        },
    }
    return out, metadata


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
        "heater_duty": {
            channel: float(result[f"resource_heater_on_{channel}"].mean())
            for channel in CHANNELS
        },
        "heater_states": sorted(
            result[["resource_heater_on_met_station_core", "resource_heater_on_laser_disdrometer"]]
            .drop_duplicates()
            .astype(int)
            .astype(str)
            .agg("".join, axis=1)
            .tolist()
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
