from __future__ import annotations

from scheduling.action_space import DiscreteActionSpace


def test_action_space_feasibility():
    space = DiscreteActionSpace(
        sensor_ids=["a", "b", "c"],
        power_costs={"a": 1.0, "b": 1.0, "c": 2.0},
        max_active=2,
        per_step_budget=2.0,
    )
    for aid in range(space.size()):
        subset = space.decode(aid)
        assert len(subset) <= 2
        assert sum(space.power_costs[s] for s in subset) <= 2.0 + 1e-12
