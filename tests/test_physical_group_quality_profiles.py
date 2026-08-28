from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _quality_builder():
    path = Path(__file__).parents[1] / "scripts" / "20_build_public_weather_truth.py"
    spec = importlib.util.spec_from_file_location("physical_quality_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.add_channel_quality_dynamics


def test_condition_dependent_quality_has_distinct_physical_group_profiles() -> None:
    builder = _quality_builder()
    frame = pd.DataFrame({
        "wind_speed_ms": [2.0, 18.0, 2.0],
        "relative_humidity": [45.0, 92.0, 45.0],
        "event_subtype_particle_latent": [0.0, 1.0, 0.0],
        "event_subtype_flux_latent": [0.0, 1.0, 0.0],
        "event_subtype_thermal_latent": [0.0, 1.0, 0.0],
    })
    sensors = (
        "gmx500_weather_station",
        "lps10_pyranometer",
        "si111_surface_ir",
        "parsivel2_disdrometer",
        "flowcapt_fc4",
    )

    out = builder(
        frame,
        sensor_ids=sensors,
        seed=7,
        coverage=0.0,
        min_duration=1,
        max_duration=1,
        min_gap=0,
        degraded_quality=0.25,
        transition_steps=0,
        report_noise_std=0.0,
        mode="condition_dependent",
    )

    assert all(f"agent_context_quality_{sensor}" in out for sensor in sensors)
    parsivel = out["agent_context_quality_parsivel2_disdrometer"].to_numpy()
    flowcapt = out["agent_context_quality_flowcapt_fc4"].to_numpy()
    assert parsivel[1] < parsivel[0]
    assert parsivel[1] < flowcapt[1]
    assert np.all((0.25 <= parsivel) & (parsivel <= 1.0))


def test_crossover_quality_uses_weather_only_physical_profiles() -> None:
    builder = _quality_builder()
    frame = pd.DataFrame({
        "wind_speed_ms": [2.0, 11.0, 18.0],
        "relative_humidity": [45.0, 82.0, 94.0],
        "air_temperature_c": [-4.0, -14.0, -24.0],
        "solar_radiation_wm2": [0.0, 160.0, 15.0],
        "event_subtype_particle_latent": [0.0, 0.0, 0.0],
        "event_subtype_flux_latent": [0.0, 0.0, 0.0],
        "event_subtype_thermal_latent": [0.0, 0.0, 0.0],
    })
    sensors = (
        "gmx500_weather_station",
        "lps10_pyranometer",
        "si111_surface_ir",
        "parsivel2_disdrometer",
        "flowcapt_fc4",
    )
    out = builder(
        frame,
        sensor_ids=sensors,
        seed=11,
        coverage=0.0,
        min_duration=1,
        max_duration=1,
        min_gap=0,
        degraded_quality=0.10,
        transition_steps=0,
        report_noise_std=0.0,
        mode="condition_dependent_crossover",
    )
    parsivel = out["agent_context_quality_parsivel2_disdrometer"].to_numpy()
    flowcapt = out["agent_context_quality_flowcapt_fc4"].to_numpy()
    pyranometer = out["agent_context_quality_lps10_pyranometer"].to_numpy()
    assert parsivel[1] > parsivel[2]
    assert flowcapt[2] > flowcapt[0]
    assert pyranometer[1] > pyranometer[0]
    assert np.all((0.10 <= out.filter(like="agent_context_quality_").to_numpy()) &
                  (out.filter(like="agent_context_quality_").to_numpy() <= 1.0))
