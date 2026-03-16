from __future__ import annotations

import itertools


class DiscreteActionSpace:
    def __init__(self, sensor_ids: list[str], power_costs: dict[str, float], max_active: int, per_step_budget: float) -> None:
        self.sensor_ids = list(sensor_ids)
        self.power_costs = dict(power_costs)
        self.max_active = int(max_active)
        self.per_step_budget = float(per_step_budget)
        self.actions = self._build_actions()
        self.action_to_id = {tuple(a): i for i, a in enumerate(self.actions)}

    def _build_actions(self) -> list[tuple[str, ...]]:
        actions: list[tuple[str, ...]] = [tuple()]
        for k in range(1, self.max_active + 1):
            for subset in itertools.combinations(self.sensor_ids, k):
                cost = sum(self.power_costs[s] for s in subset)
                if cost <= self.per_step_budget + 1e-12:
                    actions.append(tuple(subset))
        return actions

    def size(self) -> int:
        return len(self.actions)

    def decode(self, action_id: int) -> list[str]:
        return list(self.actions[action_id])

    def encode(self, subset: list[str] | tuple[str, ...]) -> int:
        key = tuple(sorted(subset, key=self.sensor_ids.index))
        if key in self.action_to_id:
            return self.action_to_id[key]
        return 0

    def nearest_feasible(self, ranked_sensor_ids: list[str]) -> int:
        for k in range(min(len(ranked_sensor_ids), self.max_active), -1, -1):
            subset = tuple(sorted(ranked_sensor_ids[:k], key=self.sensor_ids.index))
            aid = self.action_to_id.get(subset)
            if aid is not None:
                return aid
        return 0
