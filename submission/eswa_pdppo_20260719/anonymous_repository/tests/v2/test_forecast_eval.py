from __future__ import annotations

import numpy as np
import pandas as pd

from v2.evaluation import LoadedRollout
from v2.forecast_eval import forecast_loss_samples, forecast_metric_tables
from v2.oracle import LinearFrozenForecastOracle, OracleConfig, build_supervised_windows


def test_forecast_metric_tables_use_future_truth_and_conditions() -> None:
    t = np.arange(40, dtype=float)
    truth = pd.DataFrame(
        {
            "x": t,
            "wind_speed_ms": np.where(t >= 20, 9.0, 5.0),
            "air_temperature_c": np.where(t >= 30, -31.0, -20.0),
        }
    )
    observed = truth[["x", "wind_speed_ms", "air_temperature_c"]].to_numpy(dtype=float)
    masks = np.ones_like(observed)
    target = truth[["x", "wind_speed_ms"]].to_numpy(dtype=float)
    x_train, y_train = build_supervised_windows(observed, masks, target, lookback=4, horizon=2)
    oracle = LinearFrozenForecastOracle(
        OracleConfig(
            lookback=4,
            horizon=2,
            ridge_alpha=0.1,
            target_weights=(1.0, 1.0),
            target_scales=(10.0, 5.0),
        )
    ).fit(x_train, y_train)
    rollout = LoadedRollout(
        policy="demo",
        observations=observed[10:30],
        masks=masks[10:30],
        truth=observed[10:30],
        rewards=np.zeros(20),
        scores=np.zeros((20, 1)),
        powers=np.ones(20),
        peaks=np.ones(20),
        selected_masks=np.ones((20, 1), dtype=int),
        mode_ids=np.ones((20, 1), dtype=int) * 2,
        event_flags=np.zeros(20),
        oracle_losses=np.zeros(20),
        step_indices=np.arange(10, 30),
        warmup_abort_count=0,
        sensor_ids=("s",),
        state_columns=("x", "wind_speed_ms", "air_temperature_c"),
    )

    overall, by_variable, by_condition = forecast_metric_tables(
        rollout,
        truth_df=truth,
        oracle=oracle,
        metadata={"lookback": 4, "horizon": 2, "reward_target_columns": ["x", "wind_speed_ms"]},
        target_weights=(1.0, 1.0),
        target_scales=(10.0, 5.0),
    )

    assert overall["policy"] == "demo"
    assert overall["forecast_samples"] == 20
    assert overall["forecast_weighted_mae_overall"] >= 0.0
    assert np.isfinite(overall["forecast_weighted_mae_event"])
    assert len(by_variable) == 2
    assert {row["condition"] for row in by_condition} == {"all", "event", "non_event", "low_temp", "normal"}


def test_forecast_loss_samples_preserve_global_steps_and_offline_subtypes() -> None:
    t = np.arange(30, dtype=float)
    truth = pd.DataFrame(
        {
            "x": t,
            "event_subtype_id": np.where((t >= 12) & (t < 18), 2, 0),
            "blowing_snow_event": np.where((t >= 12) & (t < 18), 1, 0),
        }
    )
    observed = truth[["x"]].to_numpy(dtype=float)
    masks = np.ones_like(observed)
    x_train, y_train = build_supervised_windows(observed, masks, observed, lookback=3, horizon=2)
    oracle = LinearFrozenForecastOracle(
        OracleConfig(
            lookback=3,
            horizon=2,
            ridge_alpha=0.1,
            target_weights=(1.0,),
            target_scales=(10.0,),
            subtype_loss_weighting=True,
            subtype_flux_target_weights=(2.0,),
        )
    ).fit(x_train, y_train)
    rollout = LoadedRollout(
        policy="demo",
        observations=observed[8:20],
        masks=masks[8:20],
        truth=observed[8:20],
        rewards=np.zeros(12),
        scores=np.zeros((12, 1)),
        powers=np.ones(12),
        peaks=np.ones(12),
        selected_masks=np.ones((12, 1), dtype=int),
        mode_ids=np.ones((12, 1), dtype=int) * 2,
        event_flags=np.zeros(12),
        oracle_losses=np.zeros(12),
        step_indices=np.arange(8, 20),
        warmup_abort_count=0,
        sensor_ids=("s",),
        state_columns=("x",),
    )

    samples = forecast_loss_samples(
        rollout,
        truth_df=truth,
        oracle=oracle,
        metadata={"lookback": 3, "horizon": 2, "reward_target_columns": ["x"]},
    )

    assert samples["step_index"].tolist() == list(range(8, 20))
    assert samples.loc[samples["step_index"] == 12, "event_subtype_id"].item() == 2
    assert samples.loc[samples["step_index"] == 12, "event"].item() == 1
    assert np.all(np.isfinite(samples["forecast_loss"]))
