from __future__ import annotations

from core.config import load_yaml
from scheduling.action_space import DiscreteActionSpace
from scheduling.baselines.info_priority_scheduler import InfoPriorityScheduler
from scheduling.baselines.periodic_scheduler import PeriodicScheduler
from scheduling.baselines.random_scheduler import RandomScheduler
from scheduling.baselines.round_robin_scheduler import RoundRobinScheduler
from scheduling.baselines.warmup_round_robin_scheduler import WarmupAwareRoundRobinScheduler
from scheduling.online_projector import OnlineSubsetProjector


def _space():
    cfg = load_yaml("configs/sensors/linear_gaussian_sensors.yaml")
    sensor_ids = [s["sensor_id"] for s in cfg["sensors"]]
    power = {s["sensor_id"]: s["power_cost"] for s in cfg["sensors"]}
    return DiscreteActionSpace(sensor_ids, power, max_active=2, per_step_budget=2.2), sensor_ids


def test_baselines_valid_actions():
    space, sensor_ids = _space()
    state = {"diag_P": [1, 1, 1, 1], "freshness": [0] * len(sensor_ids), "coverage_ratio": [0] * len(sensor_ids), "previous_action": [0] * len(sensor_ids), "t": 0}

    schedulers = [
        RandomScheduler(space),
        PeriodicScheduler(space, period=1),
        RoundRobinScheduler(space, sensor_ids, max_active=2),
        InfoPriorityScheduler(space, sensor_ids, sensor_to_dims={sid: [0] for sid in sensor_ids}, max_active=2),
    ]
    for s in schedulers:
        s.reset()
        aid = s.act(state)
        assert 0 <= aid < space.size()


def test_warmup_round_robin_holds_subset_while_selected_sensor_is_warming():
    projector = OnlineSubsetProjector(
        sensor_ids=["a", "b", "c"],
        power_costs={"a": 0.2, "b": 0.3, "c": 1.3},
        startup_peak_costs={"a": 0.25, "b": 0.35, "c": 1.5},
        max_active=2,
        per_step_budget=0.8,
        startup_peak_budget=0.9,
    )
    scheduler = WarmupAwareRoundRobinScheduler(
        projector,
        sensor_ids=["a", "b", "c"],
        max_active=2,
        ready_hold_steps=1,
    )
    scheduler.reset()

    action0 = scheduler.act({"previous_action": [0.0, 0.0, 0.0], "warming_mask": [0.0, 0.0, 0.0]})
    assert action0

    state_warming = {
        "previous_action": [1.0 if sid in action0 else 0.0 for sid in ["a", "b", "c"]],
        "warming_mask": [1.0 if sid == action0[0] else 0.0 for sid in ["a", "b", "c"]],
    }
    action1 = scheduler.act(state_warming)
    assert action1 == action0
