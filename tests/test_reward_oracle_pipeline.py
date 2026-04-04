from __future__ import annotations

import numpy as np

from forecasting.dataset_builder import ForecastDataset
from forecasting.input_augmentation import augment_physical_state_series
from reward.forecast_reward import _normalize_split, _split_rollouts_then_concat


def _mk_ds(value: float, n: int = 12, lookback: int = 4, horizon: int = 2) -> ForecastDataset:
    x = np.full((n, lookback, 2), value, dtype=float)
    y = np.full((n, horizon, 1), value, dtype=float)
    return ForecastDataset(X=x, Y=y, target_indices=np.asarray([0], dtype=int))


def test_split_rollouts_then_concat_keeps_each_rollout_in_train() -> None:
    ds_a = _mk_ds(0.0)
    ds_b = _mk_ds(100.0)
    train, val, test = _split_rollouts_then_concat([ds_a, ds_b], train_ratio=0.5, val_ratio=0.25)

    assert train.X.shape[0] > 0
    assert test.X.shape[0] > 0

    train_values = set(np.unique(train.X[:, 0, 0]).tolist())
    test_values = set(np.unique(test.X[:, 0, 0]).tolist())
    # If splitting were done after concatenation, one rollout could be absent.
    assert train_values == {0.0, 100.0}
    assert test_values == {0.0, 100.0}
    assert val.X.shape[0] > 0


def test_normalize_split_constant_channel_does_not_blow_up() -> None:
    train = ForecastDataset(
        X=np.stack(
            [
                np.column_stack([np.ones(6), np.linspace(0.0, 1.0, 6)]),
                np.column_stack([np.ones(6), np.linspace(1.0, 2.0, 6)]),
            ],
            axis=0,
        ),
        Y=np.ones((2, 2, 1), dtype=float),
    )
    val = ForecastDataset(
        X=np.stack([np.column_stack([np.zeros(6), np.linspace(0.5, 1.5, 6)])], axis=0),
        Y=np.zeros((1, 2, 1), dtype=float),
    )
    test = ForecastDataset(
        X=np.stack([np.column_stack([np.zeros(6), np.linspace(0.2, 1.2, 6)])], axis=0),
        Y=np.zeros((1, 2, 1), dtype=float),
    )

    train_norm, val_norm, test_norm, stats = _normalize_split(train, val, test)
    x_std = np.asarray(stats["x_std"], dtype=float).reshape(-1)

    # Constant channel should use unit scale instead of near-zero scaling.
    assert x_std[0] == 1.0
    assert np.max(np.abs(val_norm.X)) < 10.0
    assert np.max(np.abs(test_norm.X)) < 10.0
    assert np.isfinite(train_norm.X).all()


def test_augment_physical_state_series_wraps_and_clips() -> None:
    names = [
        "wind_speed_ms",
        "wind_direction_deg",
        "air_temperature_c",
        "relative_humidity",
        "solar_radiation_wm2",
        "snow_particle_mean_diameter_mm",
        "snow_particle_mean_velocity_ms",
        "snow_mass_flux_kg_m2_s",
    ]
    series = np.asarray(
        [
            [-2.0, -30.0, -15.0, 130.0, -5.0, -0.1, -3.0, -1e-4],
            [4.0, 390.0, -14.0, -10.0, 10.0, 0.5, 2.0, 2e-4],
        ],
        dtype=float,
    )
    out, out_names, _ = augment_physical_state_series(series, names)

    idx = {name: out_names.index(name) for name in names}
    assert out[0, idx["wind_direction_deg"]] == 330.0
    assert out[1, idx["wind_direction_deg"]] == 30.0
    assert out[0, idx["wind_speed_ms"]] == 0.0
    assert out[0, idx["relative_humidity"]] == 100.0
    assert out[1, idx["relative_humidity"]] == 0.0
    assert out[0, idx["solar_radiation_wm2"]] == 0.0
    assert out[0, idx["snow_particle_mean_diameter_mm"]] == 0.0
    assert out[0, idx["snow_particle_mean_velocity_ms"]] == 0.0
    assert out[0, idx["snow_mass_flux_kg_m2_s"]] == 0.0


def test_augment_physical_state_series_adds_time_of_day_features() -> None:
    names = ["wind_speed_ms", "wind_direction_deg"]
    series = np.asarray([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]], dtype=float)
    out, out_names, _ = augment_physical_state_series(
        series,
        names,
        time_index=np.asarray([0, 21600, 43200], dtype=float),
        base_freq_s=1,
    )

    sin_idx = out_names.index("time_of_day_sin")
    cos_idx = out_names.index("time_of_day_cos")
    assert np.isclose(out[0, sin_idx], 0.0, atol=1e-6)
    assert np.isclose(out[0, cos_idx], 1.0, atol=1e-6)
    assert np.isclose(out[1, sin_idx], 1.0, atol=1e-6)
    assert np.isclose(out[2, cos_idx], -1.0, atol=1e-6)


def test_augment_physical_state_series_accepts_read_only_input() -> None:
    names = ["wind_speed_ms", "wind_direction_deg", "relative_humidity"]
    series = np.asarray([[1.0, -30.0, 120.0], [2.0, 390.0, -5.0]], dtype=float)
    series.setflags(write=False)

    out, out_names, _ = augment_physical_state_series(series, names)

    idx = {name: out_names.index(name) for name in names}
    assert out[0, idx["wind_direction_deg"]] == 330.0
    assert out[1, idx["wind_direction_deg"]] == 30.0
    assert out[0, idx["relative_humidity"]] == 100.0
    assert out[1, idx["relative_humidity"]] == 0.0
