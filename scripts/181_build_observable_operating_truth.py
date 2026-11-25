#!/usr/bin/env python3
"""Build a causal operating-factor truth view from decision-time nowcasts.

The factors are deterministic functions of nowcast weather available at the
decision epoch.  A factor at t changes target innovation at t+6 and current
measurement quality/resource demand.  Exact generator modes and event labels
are not used to build the policy-facing columns.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEAD_STEPS = 6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _shift(values: np.ndarray, lead: int) -> np.ndarray:
    out = np.empty_like(values)
    out[:-lead] = values[lead:]
    out[-lead:] = values[-1]
    return out


def _resource_builder():
    path = Path(__file__).with_name("180_build_observable_physical_resource_trace.py")
    spec = importlib.util.spec_from_file_location("observable_resource", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(
    frame: pd.DataFrame,
    *,
    use_nowcast_resource: bool = False,
    heater_quality_coupling: bool = False,
    radiometer_signal_coupling: bool = False,
) -> tuple[pd.DataFrame, dict]:
    required = {
        "agent_context_nowcast_wind_speed_ms",
        "agent_context_nowcast_air_temperature_c",
        "agent_context_nowcast_relative_humidity",
        "agent_context_nowcast_solar_radiation_wm2",
        "snow_mass_flux_kg_m2_s",
        "snow_particle_mean_velocity_ms",
        "snow_particle_mean_diameter_mm",
        "snow_surface_temperature_c",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing observable operating inputs: {missing}")

    out = frame.copy()
    wind = out["agent_context_nowcast_wind_speed_ms"].to_numpy(float)
    temp = out["agent_context_nowcast_air_temperature_c"].to_numpy(float)
    rh = out["agent_context_nowcast_relative_humidity"].to_numpy(float)
    solar = out["agent_context_nowcast_solar_radiation_wm2"].to_numpy(float)
    transport = _sigmoid((wind - 9.0) / 2.0)
    particle = _sigmoid((wind - 11.0) / 2.0)
    thermal = _sigmoid((-temp - 15.0) / 3.0) * _sigmoid((25.0 - solar) / 10.0)
    # Humidity makes the icing/thermal factor stronger without exposing a
    # latent event label or future target.
    thermal = np.clip(thermal * (0.65 + 0.35 * np.clip(rh / 100.0, 0.0, 1.0)), 0.0, 1.0)
    factors = {
        "transport": transport,
        "particle": particle,
        "thermal": thermal,
    }
    for name, values in factors.items():
        out[f"agent_context_operating_factor_{name}"] = values

    # The same decision-time factors drive the next forecast target.  The
    # scheduler sees the factors at t; target changes occur only at t+lead.
    future_transport = _shift(transport, LEAD_STEPS)
    future_particle = _shift(particle, LEAD_STEPS)
    future_thermal = _shift(thermal, LEAD_STEPS)
    out["snow_mass_flux_kg_m2_s"] = np.clip(
        out["snow_mass_flux_kg_m2_s"].to_numpy(float) + 0.0025 * future_transport,
        0.0,
        None,
    )
    out["snow_particle_mean_velocity_ms"] = np.clip(
        out["snow_particle_mean_velocity_ms"].to_numpy(float) + 5.0 * future_particle,
        0.0,
        20.0,
    )
    out["snow_particle_mean_diameter_mm"] = np.clip(
        out["snow_particle_mean_diameter_mm"].to_numpy(float) + 0.10 * future_particle,
        0.04,
        0.5,
    )
    out["snow_surface_temperature_c"] = np.clip(
        out["snow_surface_temperature_c"].to_numpy(float) + 7.0 * future_thermal,
        -80.0,
        10.0,
    )

    # Current quality is causal and sensor-specific.  Heating protects the
    # exposed channels through the resource trace; the factors determine
    # whether their measurements are relevant to the current regime.
    resource = _resource_builder().build(out, use_nowcast=use_nowcast_resource)[0]
    quality_base = {
        "met_station_core": 1.0,
        "radiometer_basic": 0.95,
        "surface_temp_ir": 0.85 + 0.15 * thermal,
        "laser_disdrometer": 0.35 + 0.65 * particle,
        "fc4_flux": 0.35 + 0.65 * transport,
        "cr1000xe_backbone": 1.0,
    }
    for channel, base in quality_base.items():
        column = f"agent_context_quality_{channel}"
        if column in out and np.isscalar(base):
            out[column] = np.clip(out[column].to_numpy(float) * float(base), 0.05, 1.0)
    if heater_quality_coupling:
        # The two exposed channels have documented heater loads.  Model the
        # physical purpose of those loads explicitly: an active heater reduces
        # icing-related measurement degradation.  The scheduler still sees
        # only the resulting quality/resource columns at decision time.
        for channel in ("met_station_core", "laser_disdrometer"):
            quality_column = f"agent_context_quality_{channel}"
            heater_column = f"resource_heater_on_{channel}"
            out[quality_column] = np.clip(
                out[quality_column].to_numpy(float)
                * (0.55 + 0.45 * resource[heater_column].to_numpy(float)),
                0.05,
                1.0,
            )
    if radiometer_signal_coupling:
        # Radiometric measurements become less informative at very low
        # irradiance.  This uses the decision-time nowcast only and leaves the
        # radiometer available when the signal is strong.
        solar = np.clip(out["agent_context_nowcast_solar_radiation_wm2"].to_numpy(float), 0.0, None)
        signal = _sigmoid((solar - 20.0) / 25.0)
        out["agent_context_quality_radiometer_basic"] = np.clip(
            out["agent_context_quality_radiometer_basic"].to_numpy(float)
            * (0.40 + 0.60 * signal),
            0.05,
            1.0,
        )
    out["agent_context_quality_surface_temp_ir"] = np.clip(
        out["agent_context_quality_surface_temp_ir"].to_numpy(float) * (0.35 + 0.65 * thermal), 0.05, 1.0
    )
    out["agent_context_quality_laser_disdrometer"] = np.clip(
        out["agent_context_quality_laser_disdrometer"].to_numpy(float) * (0.35 + 0.65 * particle), 0.05, 1.0
    )
    out["agent_context_quality_fc4_flux"] = np.clip(
        out["agent_context_quality_fc4_flux"].to_numpy(float) * (0.35 + 0.65 * transport), 0.05, 1.0
    )
    out["agent_context_quality_cr1000xe_backbone"] = 1.0
    for column in resource.columns:
        if column.startswith("resource_"):
            out[column] = resource[column].to_numpy()
    metadata = {
        "generator": Path(__file__).name,
        "lead_steps": LEAD_STEPS,
        "operating_factor_inputs": list(required - {"snow_mass_flux_kg_m2_s", "snow_particle_mean_velocity_ms", "snow_particle_mean_diameter_mm", "snow_surface_temperature_c"}),
        "target_relation": "decision-time nowcast factors drive target innovation at t+6",
        "quality_relation": "the same factors modulate specialist measurement quality",
        "resource_relation": "declared heater controller uses current weather inputs",
        "uses_exact_generator_mode": False,
        "uses_event_labels": False,
        "uses_future_targets": False,
        "uses_decision_time_nowcast_for_resource": bool(use_nowcast_resource),
        "heater_quality_coupling": bool(heater_quality_coupling),
        "radiometer_signal_coupling": bool(radiometer_signal_coupling),
        "factor_thresholds": {"transport_wind_center_ms": 9.0, "particle_wind_center_ms": 11.0, "thermal_temp_center_c": -15.0, "thermal_solar_center_wm2": 25.0},
    }
    return out, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-nowcast-resource", action="store_true")
    parser.add_argument("--heater-quality-coupling", action="store_true")
    parser.add_argument("--radiometer-signal-coupling", action="store_true")
    args = parser.parse_args()
    result, metadata = build(
        pd.read_csv(args.input),
        use_nowcast_resource=bool(args.use_nowcast_resource),
        heater_quality_coupling=bool(args.heater_quality_coupling),
        radiometer_signal_coupling=bool(args.radiometer_signal_coupling),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(result), "factor_means": {k: float(result[f"agent_context_operating_factor_{k}"].mean()) for k in ("transport", "particle", "thermal")}}, sort_keys=True))


if __name__ == "__main__":
    main()
