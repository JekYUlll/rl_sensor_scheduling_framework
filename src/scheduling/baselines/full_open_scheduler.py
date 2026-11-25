from __future__ import annotations

from scheduling.base_scheduler import BaseScheduler


class FullOpenScheduler(BaseScheduler):
    def __init__(self, sensor_ids: list[str]) -> None:
        self.sensor_ids = list(sensor_ids)

    def reset(self) -> None:
        return None

    def act(self, state: dict):
        return list(self.sensor_ids)
