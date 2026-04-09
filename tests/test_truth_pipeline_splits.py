from __future__ import annotations

import pytest

from pipelines.truth_pipeline import _RandomSubsetReplayScheduler, _split_bounds, _validate_reward_horizon_for_warmup
from scheduling.online_projector import OnlineSubsetProjector


def test_split_bounds_requires_sum_to_one() -> None:
    with pytest.raises(ValueError):
        _split_bounds(
            100,
            {
                "predictor_pretrain_ratio": 0.2,
                "rl_train_ratio": 0.5,
                "rl_val_ratio": 0.2,
                "final_test_ratio": 0.2,
            },
        )


def test_split_bounds_returns_positive_non_overlapping_segments() -> None:
    bounds = _split_bounds(
        1000,
        {
            "predictor_pretrain_ratio": 0.2,
            "rl_train_ratio": 0.5,
            "rl_val_ratio": 0.15,
            "final_test_ratio": 0.15,
        },
    )
    assert bounds["predictor_pretrain"] == (0, 200)
    assert bounds["rl_train"] == (200, 700)
    assert bounds["rl_val"] == (700, 850)
    assert bounds["final_test"] == (850, 1000)


def test_random_subset_replay_scheduler_produces_feasible_non_empty_actions() -> None:
    selector = OnlineSubsetProjector(
        sensor_ids=["a", "b", "c"],
        power_costs={"a": 0.2, "b": 0.3, "c": 1.4},
        startup_peak_costs={"a": 0.25, "b": 0.35, "c": 1.6},
        max_active=2,
        per_step_budget=0.8,
        startup_peak_budget=0.9,
    )
    scheduler = _RandomSubsetReplayScheduler(selector, seed=123, hold_min=1, hold_max=2)
    scheduler.reset()
    seen = set()
    for _ in range(12):
        subset = tuple(sorted(scheduler.act({"t": 0})))
        assert subset
        assert selector.steady_power(subset) <= selector.per_step_budget + 1e-9
        seen.add(subset)
    assert len(seen) >= 2


def test_reward_horizon_validation_rejects_warmup_longer_than_horizon() -> None:
    with pytest.raises(ValueError, match="required horizon>=6"):
        _validate_reward_horizon_for_warmup(
            {
                "sensors": [
                    {"sensor_id": "a", "warmup_steps": 0},
                    {"sensor_id": "b", "warmup_steps": 5},
                ]
            },
            {"horizon": 3},
        )
