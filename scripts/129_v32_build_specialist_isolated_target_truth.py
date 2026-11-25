"""Build the predeclared Stage-O specialist-isolated target truth."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _corrected_module():
    path = Path(__file__).with_name("120_v32_build_persistent_mode_truth_corrected.py")
    spec = importlib.util.spec_from_file_location("corrected_truth_stage_o", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ar1(rng: np.random.Generator, size: int, rho: float = 0.995) -> np.ndarray:
    innovation = rng.normal(0.0, np.sqrt(1.0 - rho * rho), size=size)
    state = np.empty(size, dtype=float)
    state[0] = innovation[0]
    for idx in range(1, size):
        state[idx] = rho * state[idx - 1] + innovation[idx]
    return np.tanh(state)


def build(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    out = _corrected_module().build_corrected(
        frame,
        seed=seed,
        forecast_lead_steps=6,
        mode_min_duration_steps=18,
        mode_max_duration_steps=36,
        forecast_noise_std=0.12,
        degradation_strength=0.65,
    )
    rng = np.random.default_rng(int(seed) + 1_912_341)
    state = _ar1(rng, len(out))
    active = out["blowing_snow_active"].astype(bool).to_numpy()
    mode = out["generator_persistent_mode_id"].to_numpy(dtype=int)
    transport = active & (mode == 0)
    particle = active & (mode == 1)
    thermal = active & (mode == 2)

    # Fixed target-range amplitudes are declared in the Stage-O design note.
    # Each innovation is written only to the corresponding specialist target.
    flux = out["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float) + 0.0040 * state * transport
    velocity = out["snow_particle_mean_velocity_ms"].to_numpy(dtype=float) + 12.0 * state * particle
    diameter = out["snow_particle_mean_diameter_mm"].to_numpy(dtype=float) + 0.25 * state * particle
    surface = out["snow_surface_temperature_c"].to_numpy(dtype=float) + 18.0 * state * thermal

    out["snow_mass_flux_kg_m2_s"] = np.where(transport, np.clip(flux, 0.0, None), out["snow_mass_flux_kg_m2_s"])
    out["snow_particle_mean_velocity_ms"] = np.where(particle, np.clip(velocity, 0.0, 20.0), out["snow_particle_mean_velocity_ms"])
    out["snow_particle_mean_diameter_mm"] = np.where(particle, np.clip(diameter, 0.04, 0.5), out["snow_particle_mean_diameter_mm"])
    out["snow_surface_temperature_c"] = np.where(thermal, np.clip(surface, -80.0, 10.0), out["snow_surface_temperature_c"])
    out["generator_independent_target_innovation"] = state
    out["generator_target_innovation_mode"] = mode
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out = build(pd.read_csv(args.input), seed=int(args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "generator": Path(__file__).name,
                "seed": int(args.seed),
                "relation": "persistent independent target innovation isolated to the matching specialist target family",
                "rho": 0.995,
                "amplitudes": {
                    "flux_kg_m2_s": 0.004,
                    "particle_velocity_ms": 12.0,
                    "particle_diameter_mm": 0.25,
                    "surface_temperature_c": 18.0,
                },
                "policy_information": "latent innovation and exact mode labels excluded; existing noisy forecast context retained",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
