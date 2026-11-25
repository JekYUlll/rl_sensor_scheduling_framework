from __future__ import annotations

import numpy as np

from scheduling.base_scheduler import BaseScheduler


class RandomScheduler(BaseScheduler):
    def __init__(self, action_space) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng()

    def reset(self) -> None:
        return None

    def act(self, state: dict) -> int:
        if hasattr(self.action_space, "sample_random_subset"):
            prev_mask = state.get("previous_action", [])
            prev_selected = [
                sid
                for sid, flag in zip(getattr(self.action_space, "sensor_ids", []), prev_mask)
                if float(flag) > 0.5
            ]
            return self.action_space.sample_random_subset(prev_selected=prev_selected, rng=self.rng)
        return int(np.random.randint(0, self.action_space.size()))
