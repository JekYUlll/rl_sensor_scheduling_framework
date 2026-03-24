from __future__ import annotations

from scheduling.base_scheduler import BaseScheduler


class PeriodicScheduler(BaseScheduler):
    def __init__(self, action_space, period: int = 1) -> None:
        self.action_space = action_space
        self.period = max(1, int(period))
        self._idx = 0
        self._last_step = -1

    def reset(self) -> None:
        self._idx = 0
        self._last_step = -1

    def act(self, state: dict) -> int | list[str]:
        t = int(state.get("t", 0))
        cycle_size = self.action_space.size()
        if hasattr(self.action_space, "project_ranked") and not hasattr(self.action_space, "decode"):
            cycle_size = max(1, len(getattr(self.action_space, "sensor_ids", [])))
        if t == 0 or (t - self._last_step) >= self.period:
            self._idx = (self._idx + 1) % cycle_size
            self._last_step = t
        if hasattr(self.action_space, "project_ranked") and not hasattr(self.action_space, "decode"):
            sensor_ids = list(getattr(self.action_space, "sensor_ids", []))
            if not sensor_ids:
                return []
            start = self._idx % len(sensor_ids)
            ranked = sensor_ids[start:] + sensor_ids[:start]
            prev_mask = state.get("previous_action", [])
            prev_selected = [sid for sid, flag in zip(sensor_ids, prev_mask) if float(flag) > 0.5]
            return self.action_space.project_ranked(ranked, prev_selected=prev_selected)
        return self._idx
