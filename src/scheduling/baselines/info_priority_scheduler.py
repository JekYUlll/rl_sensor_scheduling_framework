from __future__ import annotations

from scheduling.base_scheduler import BaseScheduler


class InfoPriorityScheduler(BaseScheduler):
    def __init__(
        self,
        action_space,
        sensor_ids: list[str],
        sensor_to_dims: dict[str, list[int]],
        max_active: int,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.action_space = action_space
        self.sensor_ids = list(sensor_ids)
        self.sensor_to_dims = dict(sensor_to_dims)
        self.max_active = int(max_active)
        self.weights = weights or {
            "uncertainty": 1.0,
            "freshness": 0.3,
            "event": 0.2,
            "coverage_deficit": 0.5,
            "switch_penalty": 0.1,
        }

    def reset(self) -> None:
        return None

    def act(self, state: dict) -> int:
        diag_p = state.get("diag_P_norm", state.get("diag_P", []))
        freshness = state.get("freshness", [0.0] * len(self.sensor_ids))
        coverage = state.get("coverage_ratio", [0.0] * len(self.sensor_ids))
        prev = state.get("previous_action", [0.0] * len(self.sensor_ids))
        event_flag = 1.0 if state.get("event", False) else 0.0

        scored = []
        for i, sid in enumerate(self.sensor_ids):
            dims = self.sensor_to_dims.get(sid, [])
            unc = sum(float(diag_p[d]) for d in dims if d < len(diag_p)) / max(1, len(dims))
            fresh = float(freshness[i]) if i < len(freshness) else 0.0
            cov = float(coverage[i]) if i < len(coverage) else 0.0
            sw = float(prev[i]) if i < len(prev) else 0.0
            score = (
                self.weights.get("uncertainty", 1.0) * unc
                + self.weights.get("freshness", 0.3) * fresh
                + self.weights.get("event", 0.2) * event_flag
                + self.weights.get("coverage_deficit", 0.5) * (1.0 - cov)
                - self.weights.get("switch_penalty", 0.1) * sw
            )
            scored.append((sid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = [sid for sid, _ in scored[: self.max_active]]
        return self.action_space.nearest_feasible(chosen)
