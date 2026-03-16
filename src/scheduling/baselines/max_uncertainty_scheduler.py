from __future__ import annotations

from scheduling.base_scheduler import BaseScheduler


class MaxUncertaintyScheduler(BaseScheduler):
    def __init__(self, action_space, sensor_ids: list[str], sensor_to_dims: dict[str, list[int]], max_active: int) -> None:
        self.action_space = action_space
        self.sensor_ids = list(sensor_ids)
        self.sensor_to_dims = dict(sensor_to_dims)
        self.max_active = int(max_active)

    def reset(self) -> None:
        return None

    def act(self, state: dict) -> int:
        diag_p = state.get("diag_P", [])
        scores = []
        for sid in self.sensor_ids:
            dims = self.sensor_to_dims.get(sid, [])
            if not dims:
                scores.append((sid, 0.0))
                continue
            score = sum(float(diag_p[d]) for d in dims if d < len(diag_p)) / max(1, len(dims))
            scores.append((sid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = [sid for sid, _ in scores[: self.max_active]]
        return self.action_space.nearest_feasible(chosen)
