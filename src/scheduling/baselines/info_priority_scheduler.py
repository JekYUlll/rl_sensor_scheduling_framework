from __future__ import annotations

from scheduling.base_scheduler import BaseScheduler


class InfoPriorityScheduler(BaseScheduler):
    """Warm-up-aware heuristic scheduler based on information priority.

    Compared with the original version, this baseline now:
    - keeps the currently selected subset while any selected sensor is warming,
      avoiding repeated wasted warm-up cycles;
    - applies sensor-specific event relevance instead of giving every sensor the
      same event bonus;
    - applies a static quality bonus so that higher-fidelity sensors observing
      the same variables are not permanently starved by ordering ties.
    """

    def __init__(
        self,
        action_space,
        sensor_ids: list[str],
        sensor_to_dims: dict[str, list[int]],
        max_active: int,
        weights: dict[str, float] | None = None,
        sensor_quality: dict[str, float] | None = None,
        sensor_target_relevance: dict[str, float] | None = None,
        sensor_event_relevance: dict[str, float] | None = None,
        hold_while_warming: bool = True,
    ) -> None:
        self.action_space = action_space
        self.sensor_ids = list(sensor_ids)
        self.sensor_to_dims = dict(sensor_to_dims)
        self.max_active = int(max_active)
        self.sensor_quality = {
            str(sid): float((sensor_quality or {}).get(sid, 0.0)) for sid in self.sensor_ids
        }
        self.sensor_target_relevance = {
            str(sid): float((sensor_target_relevance or {}).get(sid, 1.0)) for sid in self.sensor_ids
        }
        self.sensor_event_relevance = {
            str(sid): float((sensor_event_relevance or {}).get(sid, 0.0)) for sid in self.sensor_ids
        }
        self.hold_while_warming = bool(hold_while_warming)
        self.weights = weights or {
            "uncertainty": 1.0,
            "freshness": 0.3,
            "event": 0.2,
            "coverage_deficit": 0.5,
            "switch_penalty": 0.1,
            "quality": 0.35,
            "stay_on": 0.2,
        }

    def reset(self) -> None:
        return None

    def _prev_selected(self, state: dict) -> list[str]:
        prev = state.get("previous_action", [0.0] * len(self.sensor_ids))
        return [sid for sid, flag in zip(self.sensor_ids, prev) if float(flag) > 0.5]

    def _selected_warming(self, state: dict, selected: list[str]) -> bool:
        warming_mask = state.get("warming_mask", [])
        if not warming_mask:
            return False
        warming_ids = {
            sid for sid, flag in zip(self.sensor_ids, warming_mask) if float(flag) > 0.5
        }
        return any(sid in warming_ids for sid in selected)

    def _return_subset(self, subset: list[str], prev_selected: list[str]):
        if hasattr(self.action_space, "select_from_scores") and not hasattr(self.action_space, "decode"):
            return list(subset)
        return self.action_space.nearest_feasible(list(subset), prev_selected=prev_selected)

    def act(self, state: dict):
        prev = state.get("previous_action", [0.0] * len(self.sensor_ids))
        prev_selected = self._prev_selected(state)
        if self.hold_while_warming and prev_selected and self._selected_warming(state, prev_selected):
            return self._return_subset(prev_selected, prev_selected)

        diag_p = state.get("diag_P_norm", state.get("diag_P", []))
        freshness = state.get("freshness", [0.0] * len(self.sensor_ids))
        coverage = state.get("coverage_ratio", [0.0] * len(self.sensor_ids))
        ready_mask = state.get("ready_mask", [0.0] * len(self.sensor_ids))
        event_flag = 1.0 if state.get("event", False) else 0.0

        scored = []
        for i, sid in enumerate(self.sensor_ids):
            dims = self.sensor_to_dims.get(sid, [])
            unc = sum(float(diag_p[d]) for d in dims if d < len(diag_p)) / max(1, len(dims))
            fresh = float(freshness[i]) if i < len(freshness) else 0.0
            cov = float(coverage[i]) if i < len(coverage) else 0.0
            sw = float(prev[i]) if i < len(prev) else 0.0
            ready = float(ready_mask[i]) if i < len(ready_mask) else 0.0
            target_rel = self.sensor_target_relevance.get(sid, 1.0)
            event_rel = self.sensor_event_relevance.get(sid, 0.0)
            quality = self.sensor_quality.get(sid, 0.0)
            info_score = (
                self.weights.get("uncertainty", 1.0) * unc
                + self.weights.get("quality", 0.0) * unc * quality
                + self.weights.get("freshness", 0.3) * fresh
                + self.weights.get("coverage_deficit", 0.5) * (1.0 - cov)
            )
            score = (
                target_rel * info_score
                + self.weights.get("event", 0.2) * event_flag * event_rel
                + self.weights.get("stay_on", 0.0) * sw * max(ready, 0.5)
                - self.weights.get("switch_penalty", 0.1) * (1.0 - sw)
            )
            scored.append((sid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if hasattr(self.action_space, "select_from_scores") and not hasattr(self.action_space, "decode"):
            return self.action_space.select_from_scores(
                {sid: score for sid, score in scored},
                prev_selected=prev_selected,
            )
        chosen = [sid for sid, _ in scored[: self.max_active]]
        return self.action_space.nearest_feasible(chosen, prev_selected=prev_selected)
