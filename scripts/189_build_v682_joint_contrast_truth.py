#!/usr/bin/env python3
"""Build V682 with a predeclared specialist target/quality contrast.

This is a synthetic scenario calibration, not a final-test fit.  The causal
mode and nowcast structure from V527-r2 is retained.  The change is limited to
two physically interpretable effects: event-specific target excursions are
larger, and the specialist associated with the current operating mode remains
usable while exposed alternatives suffer lower observation quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODE_TO_SENSOR = {0: "fc4_flux", 1: "laser_disdrometer", 2: "surface_temp_ir"}
MODE_NAMES = ("transport", "particle", "thermal")


def build(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "generator_persistent_mode_id",
        "generator_persistent_mode_strength",
        "generator_observable_wind_driver",
        "generator_observable_thermal_driver",
        "snow_mass_flux_kg_m2_s",
        "snow_particle_mean_velocity_ms",
        "snow_particle_mean_diameter_mm",
        "snow_surface_temperature_c",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"truth is missing V682 inputs: {missing}")
    out = frame.copy()
    mode = out["generator_persistent_mode_id"].to_numpy(dtype=int)
    strength = np.clip(out["generator_persistent_mode_strength"].to_numpy(dtype=float), 0.0, 1.0)
    wind = np.clip((out["generator_observable_wind_driver"].to_numpy(dtype=float) + 1.0) / 2.0, 0.0, 1.0)
    cold = np.clip((-out["generator_observable_thermal_driver"].to_numpy(dtype=float) + 1.0) / 2.0, 0.0, 1.0)

    # Fixed amplitudes were chosen before any V682 geometry result.  They keep
    # the variables in the existing synthetic cold-region ranges while making
    # specialist-specific future value measurable.
    transport = mode == 0
    particle = mode == 1
    thermal = mode == 2
    out["snow_mass_flux_kg_m2_s"] = np.where(
        transport,
        np.clip(out["snow_mass_flux_kg_m2_s"].to_numpy(float) + 0.0030 * wind * strength, 0.0, None),
        out["snow_mass_flux_kg_m2_s"],
    )
    out["snow_particle_mean_velocity_ms"] = np.where(
        particle,
        np.clip(out["snow_particle_mean_velocity_ms"].to_numpy(float) + 10.0 * wind * strength, 0.0, 20.0),
        out["snow_particle_mean_velocity_ms"],
    )
    out["snow_particle_mean_diameter_mm"] = np.where(
        particle,
        np.clip(out["snow_particle_mean_diameter_mm"].to_numpy(float) + 0.18 * wind * strength, 0.04, 0.5),
        out["snow_particle_mean_diameter_mm"],
    )
    out["snow_surface_temperature_c"] = np.where(
        thermal,
        np.clip(out["snow_surface_temperature_c"].to_numpy(float) + 15.0 * cold * strength, -80.0, 10.0),
        out["snow_surface_temperature_c"],
    )

    # Quality scores are noise-reduction scores.  The mode-matched specialist
    # remains usable; exposed alternatives degrade with the same causal mode
    # strength. This uses latent mode only to generate truth-side quality, not
    # as a policy input.
    for mode_id, sensor in MODE_TO_SENSOR.items():
        column = f"agent_context_quality_{sensor}"
        if column not in out:
            raise ValueError(f"truth is missing quality column: {column}")
        base = out[column].to_numpy(dtype=float)
        matched = mode == mode_id
        out[column] = np.where(
            matched,
            np.clip(0.92 + 0.06 * strength, 0.05, 1.0),
            np.clip(base * (1.0 - 0.65 * strength), 0.05, 1.0),
        )

    return out, {
        "generator": Path(__file__).name,
        "parent_truth": "v527_observable_target_truth_20260909_r2",
        "resource_controller": "v681_v527_observable_mode_proxy_wind_cold_hysteresis",
        "mode_names": list(MODE_NAMES),
        "target_amplitudes": {
            "flux_kg_m2_s": 0.0030,
            "particle_velocity_ms": 10.0,
            "particle_diameter_mm": 0.18,
            "surface_temperature_c": 15.0,
        },
        "quality_relation": {
            "matched_quality": "0.92 + 0.06*mode_strength",
            "unmatched_multiplier": "1 - 0.65*mode_strength",
        },
        "policy_information": "existing V527 online proxies retained; exact mode and target innovation excluded",
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
    print(json.dumps({"rows": len(result), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
