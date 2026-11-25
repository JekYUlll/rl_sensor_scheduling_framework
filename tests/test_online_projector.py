from __future__ import annotations

import numpy as np

from scheduling.online_projector import OnlineSubsetProjector


def test_random_subset_defaults_to_non_empty_when_possible() -> None:
    projector = OnlineSubsetProjector(
        sensor_ids=["a", "b", "c"],
        power_costs={"a": 1.0, "b": 1.0, "c": 1.0},
        max_active=2,
        per_step_budget=2.0,
    )
    rng = np.random.default_rng(42)
    for _ in range(32):
        subset = projector.sample_random_subset(prev_selected=[], rng=rng)
        assert len(subset) >= 1


def test_random_subset_allows_empty_when_requested() -> None:
    projector = OnlineSubsetProjector(
        sensor_ids=["a", "b"],
        power_costs={"a": 1.0, "b": 1.0},
        max_active=1,
        per_step_budget=1.0,
    )
    rng = np.random.default_rng(0)
    seen_empty = False
    for _ in range(128):
        subset = projector.sample_random_subset(prev_selected=[], rng=rng, allow_empty=True)
        if len(subset) == 0:
            seen_empty = True
            break
    assert seen_empty
