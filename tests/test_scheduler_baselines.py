from __future__ import annotations

from core.config import load_yaml
from scheduling.action_space import DiscreteActionSpace
from scheduling.baselines.info_priority_scheduler import InfoPriorityScheduler
from scheduling.baselines.periodic_scheduler import PeriodicScheduler
from scheduling.baselines.random_scheduler import RandomScheduler
from scheduling.baselines.round_robin_scheduler import RoundRobinScheduler


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
