from __future__ import annotations

from scheduling.base_scheduler import BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    def __init__(self, action_space, sensor_ids: list[str], max_active: int, min_on_steps: int = 1) -> None:
        self.action_space = action_space
        self.sensor_ids = list(sensor_ids)
        self.max_active = int(max_active)
        self.min_on_steps = max(1, int(min_on_steps))
        self.ptr = 0
        self.hold = 0
        self.cached_action = 0

    def reset(self) -> None:
        self.ptr = 0
        self.hold = 0
        self.cached_action = 0

    def act(self, state: dict) -> int:
        if self.hold > 0:
            self.hold -= 1
            return self.cached_action
        chosen = []
        for i in range(self.max_active):
            chosen.append(self.sensor_ids[(self.ptr + i) % len(self.sensor_ids)])
        self.ptr = (self.ptr + self.max_active) % len(self.sensor_ids)
        self.cached_action = self.action_space.nearest_feasible(chosen)
        self.hold = self.min_on_steps - 1
        return self.cached_action
