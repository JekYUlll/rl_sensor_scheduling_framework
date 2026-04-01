from __future__ import annotations

import numpy as np

from reward.forecast_reward import FrozenForecastRewardOracle
from reward.mainline_reward import compute_forecast_task_terms, load_training_reward_cfg


class _ZeroPredictor:
    def predict(self, ds):
        return np.zeros_like(ds.Y, dtype=np.float32)


def test_horizon_weights_are_applied_in_forecast_oracle() -> None:
    oracle = FrozenForecastRewardOracle(
        predictor=_ZeroPredictor(),
        lookback=2,
        horizon=3,
        base_feature_names=["x0"],
        input_columns=["x0"],
        target_columns=["x0"],
        x_mean=np.zeros((1, 1, 1), dtype=np.float32),
        x_std=np.ones((1, 1, 1), dtype=np.float32),
        y_mean=np.zeros((1, 1, 1), dtype=np.float32),
        y_std=np.ones((1, 1, 1), dtype=np.float32),
        loss_name="mse",
        loss_delta=1.0,
        use_observed_mask=False,
        use_time_delta=False,
        horizon_weights=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        score_scale=1.0,
        score_clip=100.0,
    )
    history = np.asarray([[1.0], [2.0]], dtype=float)
    future = np.asarray([[1.0], [3.0], [5.0]], dtype=float)
    score = oracle.score(history, future, None)
    assert np.isclose(score, 9.0)


def test_mainline_reward_is_forecast_dominated_with_explicit_penalties() -> None:
    cfg = load_training_reward_cfg(
        {
            "reward": {
                "lambda_forecast": 1.0,
                "lambda_switch": 0.5,
                "lambda_coverage": 2.0,
                "lambda_violation": 3.0,
                "lambda_state_tracking": 0.0,
                "min_coverage_ratio": 0.5,
            },
            "constraints": {"min_coverage_ratio": 0.5},
        }
    )
    terms = compute_forecast_task_terms(
        forecast_loss=2.0,
        switch_count=2,
        coverage_ratio=[0.5, 0.0],
        steady_power=2.4,
        peak_power=2.6,
        steady_limit=2.0,
        peak_limit=2.5,
        reward_cfg=cfg,
    )
    expected_coverage = 0.5
    expected_violation = (0.4 / 2.0) + (0.1 / 2.5)
    expected_loss = 2.0 + 0.5 * 2.0 + 2.0 * expected_coverage + 3.0 * expected_violation
    assert np.isclose(terms["coverage_penalty_raw"], expected_coverage)
    assert np.isclose(terms["violation_penalty_raw"], expected_violation)
    assert np.isclose(terms["task_loss"], expected_loss)
    assert np.isclose(terms["task_reward"], -expected_loss)
