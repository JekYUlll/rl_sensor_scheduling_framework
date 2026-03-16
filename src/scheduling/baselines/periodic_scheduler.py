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

    def act(self, state: dict) -> int:
        t = int(state.get("t", 0))
        if t == 0 or (t - self._last_step) >= self.period:
            self._idx = (self._idx + 1) % self.action_space.size()
            self._last_step = t
        return self._idx
