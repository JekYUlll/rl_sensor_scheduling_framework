from __future__ import annotations

import numpy as np

from scheduling.base_scheduler import BaseScheduler


class RandomScheduler(BaseScheduler):
    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def reset(self) -> None:
        return None

    def act(self, state: dict) -> int:
        return int(np.random.randint(0, self.action_space.size()))
