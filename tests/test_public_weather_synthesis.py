from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_sources.public_weather_synthesis import (
    PublicWeatherSynthesisConfig,
    STATE_COLUMNS,
    _assign_event_subtypes,
    _intensity_conditioned_context_signal,
    generate_public_weather_truth,
    load_antaws_station,
    validate_synthetic_against_anchor,
)


def test_intensity_conditioned_context_signal_is_bounded_and_optional() -> None:
    base = np.asarray([0.0, 0.5, 1.0, 1.0, 0.0])
    latent = np.asarray([0.0, 0.0, 0.2, 2.0, 0.0])
    mask = np.asarray([False, False, True, True, False])

    np.testing.assert_allclose(
        _intensity_conditioned_context_signal(base, latent, mask, lead_steps=0, strength=0.0),
        base,
    )
    conditioned = _intensity_conditioned_context_signal(
        base, latent, mask, lead_steps=0, strength=1.0
    )
    assert np.all((0.0 <= conditioned) & (conditioned <= 1.0))
    assert conditioned[3] > conditioned[2] > 0.0


def test_stratified_event_subtypes_control_run_counts() -> None:
    active = np.tile(np.asarray([True, True, False]), 12)
    subtype = _assign_event_subtypes(
        active,
        rng=np.random.default_rng(7),
        particle_prob=0.5,
        flux_prob=0.25,
        thermal_prob=0.25,
        assignment="stratified",
    )
    run_labels = subtype[np.arange(0, active.size, 3)]
    assert np.bincount(run_labels, minlength=4)[1:].tolist() == [6, 3, 3]


def test_particle_subtype_respects_run_level_sensor_availability() -> None:
    active = np.tile(np.asarray([True, True, False]), 12)
    eligible = np.tile(np.asarray([False, False, True]), 12)
    eligible[18:20] = True
    subtype = _assign_event_subtypes(
        active,
        rng=np.random.default_rng(7),
        particle_prob=0.5,
        flux_prob=0.25,
        thermal_prob=0.25,
        assignment="stratified",
        particle_eligibility=eligible,
        particle_min_eligibility_fraction=0.8,
    )
    run_labels = subtype[np.arange(0, active.size, 3)]
    assert np.count_nonzero(run_labels == 1) == 1
    assert run_labels[6] == 1


def _write_antaws_station(root: Path, station: str, rows: int = 16) -> None:
    root.mkdir(parents=True, exist_ok=True)
    data = []
    for idx in range(rows):
        hour = (idx * 3) % 24
        day = 1 + (idx * 3) // 24
        data.append(
            {
                "Year": 2020,
                "Month": 1,
                "Day": day,
                "Three-hourly observation time(UTC)": hour,
                "Temperature()": -20.0 + 0.1 * idx,
                "Pressure(hPa)": 700.0 + 0.2 * idx,
                "Wind Speed(m/s)": 7.0 + 0.3 * idx,
                "Wind Direction": (30.0 + 4.0 * idx) % 360.0,
                "Relative Humidity(%)": 60.0 + 0.1 * idx,
            }
        )
    pd.DataFrame(data).to_csv(root / f"{station}_3h.csv", index=False)


def test_load_antaws_station_normalizes_columns(tmp_path: Path) -> None:
    _write_antaws_station(tmp_path, "Demo")
    df = load_antaws_station(tmp_path, "Demo")

    assert "timestamp" in df.columns
    assert "air_temperature_c" in df.columns
    assert "air_pressure_pa" in df.columns
    assert "wind_dir_sin" in df.columns
    assert len(df) == 16


def test_generate_public_weather_truth_is_training_compatible(tmp_path: Path) -> None:
    _write_antaws_station(tmp_path, "DemoA", rows=24)
    _write_antaws_station(tmp_path, "DemoB", rows=24)
    cfg = PublicWeatherSynthesisConfig(
        antaws_root=tmp_path,
        stations=("DemoA", "DemoB"),
        steps=32,
        freq_s=10800,
        seed=7,
    )

    df, meta = generate_public_weather_truth(cfg)

    assert len(df) == 32
    assert set(STATE_COLUMNS).issubset(df.columns)
    assert df[STATE_COLUMNS].isna().sum().sum() == 0
    assert "parsivel_available" in df.columns
    assert "blowing_snow_active" in df.columns
    assert 0.20 <= float(df["event_flag"].mean()) <= 0.80
    assert meta["steps"] == 32
    assert "blowing_snow_event_coverage_actual" in meta


def test_validation_report_contains_core_variables(tmp_path: Path) -> None:
    _write_antaws_station(tmp_path, "Demo", rows=24)
    cfg = PublicWeatherSynthesisConfig(antaws_root=tmp_path, stations=("Demo",), steps=24, freq_s=10800)
    df, _ = generate_public_weather_truth(cfg)
    anchor = load_antaws_station(tmp_path, "Demo")
    anchor["air_pressure_pa"] = anchor["air_pressure_hpa"] * 100.0

    report = validate_synthetic_against_anchor(anchor, df, max_lag=3)

    assert "air_temperature_c" in set(report["variable"])
    assert "wind_speed_ms" in set(report["variable"])
