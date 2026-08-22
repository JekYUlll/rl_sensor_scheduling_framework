from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_sources.public_weather_synthesis import (
    PublicWeatherSynthesisConfig,
    STATE_COLUMNS,
    generate_public_weather_truth,
    load_antaws_station,
    validate_synthetic_against_anchor,
)


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
    assert df["particle_spectrum_precursor"].equals(df["event_subtype_particle_latent"])
    assert df["flux_burst_precursor"].equals(df["event_subtype_flux_latent"])
    assert df["surface_thermal_gradient_precursor"].equals(
        df["event_subtype_thermal_latent"]
    )


def test_validation_report_contains_core_variables(tmp_path: Path) -> None:
    _write_antaws_station(tmp_path, "Demo", rows=24)
    cfg = PublicWeatherSynthesisConfig(antaws_root=tmp_path, stations=("Demo",), steps=24, freq_s=10800)
    df, _ = generate_public_weather_truth(cfg)
    anchor = load_antaws_station(tmp_path, "Demo")
    anchor["air_pressure_pa"] = anchor["air_pressure_hpa"] * 100.0

    report = validate_synthetic_against_anchor(anchor, df, max_lag=3)

    assert "air_temperature_c" in set(report["variable"])
    assert "wind_speed_ms" in set(report["variable"])
