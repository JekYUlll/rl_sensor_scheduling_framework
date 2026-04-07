from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from forecasting.input_augmentation import augment_input_series
from forecasting.series_preparation import prepare_input_and_targets
from reward.forecast_reward import FrozenForecastRewardOracle, _resolve_manifest_artifact_path


class _RecordingPredictor:
    def __init__(self) -> None:
        self.last_x: np.ndarray | None = None

    def predict(self, ds):
        self.last_x = np.asarray(ds.X, dtype=np.float32)
        return np.zeros_like(ds.Y, dtype=np.float32)


def test_augment_input_series_appends_context_before_mask_and_delta() -> None:
    series = np.asarray([[1.0], [2.0], [3.0]], dtype=float)
    observed_mask = np.asarray([[1.0], [0.0], [1.0]], dtype=float)
    augmented, names = augment_input_series(
        input_series=series,
        observed_mask=observed_mask,
        feature_names=["wind_speed_ms"],
        use_observed_mask=True,
        use_time_delta=True,
        context_series={
            "trace_p": np.asarray([0.2, 0.3, 0.4], dtype=float),
            "event_flags": np.asarray([0.0, 0.0, 1.0], dtype=float),
        },
        context_features=["trace_p", "event_transition"],
    )

    assert names == [
        "wind_speed_ms",
        "trace_p",
        "event_transition",
        "is_observed_wind_speed_ms",
        "delta_wind_speed_ms",
    ]
    np.testing.assert_allclose(augmented[:, 1], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(augmented[:, 2], [0.0, 0.0, 1.0])
    assert "is_observed_trace_p" not in names
    assert "delta_trace_p" not in names


def test_prepare_input_and_targets_keeps_targets_and_adds_context() -> None:
    input_series = np.asarray([[10.0], [11.0], [12.0]], dtype=float)
    target_series = np.asarray([[20.0], [21.0], [22.0]], dtype=float)
    prepared_x, input_names, prepared_y, target_names, target_indices = prepare_input_and_targets(
        input_series=input_series,
        target_series=target_series,
        feature_names=["air_temperature_c"],
        observed_mask=np.ones_like(input_series, dtype=float),
        use_observed_mask=False,
        use_time_delta=False,
        target_columns=["air_temperature_c"],
        context_series={"trace_p": np.asarray([0.5, 0.4, 0.3], dtype=float)},
        context_features=["trace_p"],
    )

    assert input_names == ["air_temperature_c", "trace_p"]
    assert target_names == ["air_temperature_c"]
    np.testing.assert_array_equal(target_indices, np.asarray([0], dtype=int))
    np.testing.assert_allclose(prepared_x[:, 1], [0.5, 0.4, 0.3])
    np.testing.assert_allclose(prepared_y[:, 0], [20.0, 21.0, 22.0])


def test_reward_oracle_score_consumes_context_features() -> None:
    predictor = _RecordingPredictor()
    oracle = FrozenForecastRewardOracle(
        predictor=predictor,
        lookback=2,
        horizon=1,
        base_feature_names=["x0"],
        input_columns=["x0", "trace_p", "event_flag"],
        target_columns=["x0"],
        x_mean=np.zeros((1, 1, 3), dtype=np.float32),
        x_std=np.ones((1, 1, 3), dtype=np.float32),
        y_mean=np.zeros((1, 1, 1), dtype=np.float32),
        y_std=np.ones((1, 1, 1), dtype=np.float32),
        loss_name="mse",
        loss_delta=1.0,
        use_observed_mask=False,
        use_time_delta=False,
        horizon_weights=np.asarray([1.0], dtype=np.float32),
        context_features=["trace_p", "event_flag"],
        score_scale=1.0,
        score_clip=100.0,
    )

    score = oracle.score(
        history_window=np.asarray([[1.0], [2.0]], dtype=float),
        future_truth=np.asarray([[3.0]], dtype=float),
        context_series_window={
            "trace_p": np.asarray([0.25, 0.5], dtype=float),
            "event_flags": np.asarray([0.0, 1.0], dtype=float),
        },
    )

    assert np.isclose(score, 9.0)
    assert predictor.last_x is not None
    assert predictor.last_x.shape == (1, 2, 3)
    np.testing.assert_allclose(predictor.last_x[0, :, 1], [0.25, 0.5])
    np.testing.assert_allclose(predictor.last_x[0, :, 2], [0.0, 1.0])


def test_manifest_artifact_path_falls_back_to_local_copy(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_dir = Path(tmpdir)
        local_artifact = manifest_dir / "reward_predictor_tcn.pt"
        local_artifact.write_bytes(b"placeholder")
        inaccessible = Path(
            "/root/autodl-tmp/microclimate_demo/rl_sensor_scheduling_framework/reports/runs/demo/reward_predictor_tcn.pt"
        )

        original_exists = Path.exists

        def guarded_exists(path: Path) -> bool:
            if path == inaccessible:
                raise PermissionError("permission denied for remote artifact path")
            return original_exists(path)

        monkeypatch.setattr(Path, "exists", guarded_exists)

        resolved = _resolve_manifest_artifact_path(
            manifest_dir,
            str(inaccessible),
        )

        assert resolved == local_artifact
