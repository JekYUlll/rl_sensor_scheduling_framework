#!/usr/bin/env python3
"""Build a truth-only scene with shared observable weather drivers.

The same nowcast-derived drivers determine future target innovation,
channel-quality regimes, and the documented heater demands.  The driver
construction uses only quantities available at the scheduling epoch; exact
future targets and latent diagnostic columns are never policy inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEAD_STEPS = 6
CHANNELS = ("fc4_flux", "laser_disdrometer", "surface_temp_ir", "radiometer_basic")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _lagged(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    if lag <= 0:
        return values.copy()
    out[lag:] = values[:-lag]
    return out


def build(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "time_idx",
        "snow_mass_flux_kg_m2_s",
        "snow_particle_mean_velocity_ms",
        "snow_particle_mean_diameter_mm",
        "snow_surface_temperature_c",
        "agent_context_nowcast_wind_speed_ms",
        "agent_context_nowcast_relative_humidity",
        "agent_context_nowcast_air_temperature_c",
        "agent_context_nowcast_solar_radiation_wm2",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"truth is missing required columns: {missing}")
    out = frame.copy()
    wind = out["agent_context_nowcast_wind_speed_ms"].to_numpy(float)
    rh = out["agent_context_nowcast_relative_humidity"].to_numpy(float)
    air = out["agent_context_nowcast_air_temperature_c"].to_numpy(float)
    solar = out["agent_context_nowcast_solar_radiation_wm2"].to_numpy(float)

    # These bounded scores are declared observable drivers, not post-hoc labels.
    transport = _sigmoid((wind - 10.0) / 2.0) * _sigmoid((rh - 66.0) / 4.0)
    particle = _sigmoid((wind - 12.0) / 2.5) * (0.55 + 0.45 * _sigmoid((rh - 62.0) / 5.0))
    icing = _sigmoid((-air - 18.0) / 5.0) * _sigmoid((rh - 72.0) / 5.0)
    thermal = np.clip(0.65 * icing + 0.35 * _sigmoid((-solar - 5.0) / 20.0), 0.0, 1.0)
    drivers = {"transport": transport, "particle": particle, "thermal": thermal}
    for name, values in drivers.items():
        out[f"generator_shared_{name}_driver"] = values
        out[f"agent_context_shared_{name}_driver"] = values

    # A six-step lead relation: at time t, the available nowcast at t-6 is
    # the declared driver for the current target and quality state.
    lead = {name: _lagged(values, LEAD_STEPS) for name, values in drivers.items()}
    flux = out["snow_mass_flux_kg_m2_s"].to_numpy(float)
    velocity = out["snow_particle_mean_velocity_ms"].to_numpy(float)
    diameter = out["snow_particle_mean_diameter_mm"].to_numpy(float)
    surface = out["snow_surface_temperature_c"].to_numpy(float)
    out["snow_mass_flux_kg_m2_s"] = np.clip(flux * (1.0 + 0.55 * lead["transport"]), 0.0, None)
    out["snow_particle_mean_velocity_ms"] = np.clip(velocity + 4.0 * lead["particle"], 0.0, 20.0)
    out["snow_particle_mean_diameter_mm"] = np.clip(diameter + 0.075 * lead["particle"], 0.04, 0.5)
    out["snow_surface_temperature_c"] = np.clip(surface + 5.0 * lead["thermal"], -80.0, 10.0)

    # Quality scores follow the same lead drivers.  The matching specialist
    # remains reliable; nonmatching channels move toward the declared 0.10
    # floor, which represents a large noise multiplier in the simulator.
    affinities = {
        "fc4_flux": lead["transport"],
        "laser_disdrometer": lead["particle"],
        "surface_temp_ir": lead["thermal"],
        "radiometer_basic": lead["thermal"],
    }
    for sensor in CHANNELS:
        column = f"agent_context_quality_{sensor}"
        if column not in out:
            continue
        baseline = out[column].to_numpy(float)
        affinity = affinities[sensor]
        floor = 0.10 if sensor != "radiometer_basic" else 0.20
        out[column] = np.clip(baseline * (floor + (1.0 - floor) * affinity), 0.05, 1.0)

    # Truth-side diagnostics make the causal chain auditable without exposing
    # any exact state label to the scheduler.
    driver_names = np.asarray(tuple(drivers), dtype=object)
    out["generator_shared_dominant_driver"] = driver_names[
        np.argmax(np.vstack(tuple(drivers.values())), axis=0)
    ]
    out["generator_shared_icing_driver"] = icing
    out["generator_shared_transport_driver"] = transport
    out["generator_shared_particle_driver"] = particle
    out["generator_shared_thermal_driver"] = thermal
    return out, {
        "generator": Path(__file__).name,
        "lead_steps": LEAD_STEPS,
        "online_driver_columns": [f"agent_context_shared_{name}_driver" for name in drivers],
        "quality_columns": [f"agent_context_quality_{name}" for name in CHANNELS],
        "resource_controller_inputs": [
            "agent_context_nowcast_wind_speed_ms",
            "agent_context_nowcast_relative_humidity",
            "agent_context_nowcast_air_temperature_c",
            "agent_context_nowcast_solar_radiation_wm2",
        ],
        "exact_labels_policy_input": False,
        "target_relation": "six-step lagged shared nowcast drivers",
        "quality_relation": "matching specialist affinity with declared floors",
        "target_amplitudes": {"flux_multiplier": 0.55, "particle_velocity_ms": 4.0, "particle_diameter_mm": 0.075, "thermal_surface_c": 5.0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, metadata = build(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(result),
        "driver_support": {name: float(result[f"generator_shared_{name}_driver"].mean()) for name in ("transport", "particle", "thermal")},
        "dominant_counts": result["generator_shared_dominant_driver"].value_counts().to_dict(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
