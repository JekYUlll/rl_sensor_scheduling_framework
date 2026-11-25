from __future__ import annotations

from scheduling.base_scheduler import BaseScheduler


class WarmupAwareRoundRobinScheduler(BaseScheduler):
    """Round-robin baseline that avoids wasting warm-up progress.

    The scheduler keeps the current subset while any selected sensor is still
    warming. Once the subset is fully ready, it holds for a few extra steps
    before rotating to the next ranked subset.
    """

    def __init__(
        self,
        action_space,
        sensor_ids: list[str],
        max_active: int,
        ready_hold_steps: int = 2,
    ) -> None:
        self.action_space = action_space
        self.sensor_ids = list(sensor_ids)
        self.max_active = int(max_active)
        self.ready_hold_steps = max(0, int(ready_hold_steps))
        self.ptr = 0
        self.current_subset: list[str] = []
        self.ready_hold_remaining = 0

    def reset(self) -> None:
        self.ptr = 0
        self.current_subset = []
        self.ready_hold_remaining = 0

    def _prev_selected(self, state: dict) -> list[str]:
        prev_mask = state.get("previous_action", [])
        return [sid for sid, flag in zip(self.sensor_ids, prev_mask) if float(flag) > 0.5]

    def _selected_warming(self, state: dict, selected: list[str]) -> bool:
        warming_mask = state.get("warming_mask", [])
        if not warming_mask:
            return False
        warming_ids = {
            sid for sid, flag in zip(self.sensor_ids, warming_mask) if float(flag) > 0.5
        }
        return any(sid in warming_ids for sid in selected)

    def _ranked_subset(self, prev_selected: list[str]):
        if not self.sensor_ids:
            return [] if hasattr(self.action_space, "project_ranked") else 0
        limit = self.max_active
        step_stride = self.max_active
        if hasattr(self.action_space, "project_ranked") and not hasattr(self.action_space, "decode"):
            limit = len(self.sensor_ids)
            step_stride = 1
        ranked = [self.sensor_ids[(self.ptr + i) % len(self.sensor_ids)] for i in range(limit)]
        self.ptr = (self.ptr + step_stride) % len(self.sensor_ids)
        if hasattr(self.action_space, "project_ranked") and not hasattr(self.action_space, "decode"):
            return self.action_space.project_ranked(ranked, prev_selected=prev_selected)
        action_id = self.action_space.nearest_feasible(ranked, prev_selected=prev_selected)
        return action_id

    def act(self, state: dict):
        prev_selected = self._prev_selected(state)
        if prev_selected:
            self.current_subset = list(prev_selected)
            if self._selected_warming(state, prev_selected):
                if hasattr(self.action_space, "project_ranked") and not hasattr(self.action_space, "decode"):
                    return list(prev_selected)
                return self.action_space.encode(prev_selected)
            if self.ready_hold_remaining > 0:
                self.ready_hold_remaining -= 1
                if hasattr(self.action_space, "project_ranked") and not hasattr(self.action_space, "decode"):
                    return list(prev_selected)
                return self.action_space.encode(prev_selected)
        action = self._ranked_subset(prev_selected)
        if hasattr(self.action_space, "project_ranked") and not hasattr(self.action_space, "decode"):
            self.current_subset = list(action)
        else:
            self.current_subset = self.action_space.decode(int(action))
        self.ready_hold_remaining = self.ready_hold_steps
        return action
