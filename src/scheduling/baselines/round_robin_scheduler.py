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
        limit = self.max_active
        step_stride = self.max_active
        if hasattr(self.action_space, "project_ranked") and not hasattr(self.action_space, "decode"):
            limit = len(self.sensor_ids)
            # In projector mode we rotate a full ranking over all sensors.
            # Advancing by max_active would stall whenever max_active == len(sensor_ids).
            step_stride = 1
        for i in range(limit):
            chosen.append(self.sensor_ids[(self.ptr + i) % len(self.sensor_ids)])
        self.ptr = (self.ptr + step_stride) % len(self.sensor_ids)
        if hasattr(self.action_space, "project_ranked") and not hasattr(self.action_space, "decode"):
            prev_mask = state.get("previous_action", [])
            prev_selected = [sid for sid, flag in zip(self.sensor_ids, prev_mask) if float(flag) > 0.5]
            self.cached_action = self.action_space.project_ranked(chosen, prev_selected=prev_selected)
        else:
            self.cached_action = self.action_space.nearest_feasible(chosen)
        self.hold = self.min_on_steps - 1
        return self.cached_action
