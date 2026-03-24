from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np


class DiscreteActionSpace:
    def __init__(
        self,
        sensor_ids: list[str],
        power_costs: dict[str, float],
        max_active: int,
        per_step_budget: float,
        startup_peak_costs: dict[str, float] | None = None,
        startup_peak_budget: float | None = None,
        safety_margin: float = 0.0,
        required_sensor_ids: list[str] | None = None,
    ) -> None:
        self.sensor_ids = list(sensor_ids)
        self.power_costs = {str(k): float(v) for k, v in power_costs.items()}
        self.startup_peak_costs = {
            sid: float((startup_peak_costs or {}).get(sid, self.power_costs[sid])) for sid in self.sensor_ids
        }
        self.max_active = int(max_active)
        self.required_sensor_ids = [str(sid) for sid in (required_sensor_ids or [])]
        missing_required = [sid for sid in self.required_sensor_ids if sid not in self.sensor_ids]
        if missing_required:
            raise ValueError(f"required sensors not present in action space: {missing_required}")
        if len(self.required_sensor_ids) > self.max_active:
            raise ValueError("number of required sensors exceeds max_active")
        self.per_step_budget = float(per_step_budget)
        self.startup_peak_budget = None if startup_peak_budget is None else float(startup_peak_budget)
        self.safety_margin = max(0.0, float(safety_margin))
        self.actions = self._build_actions()
        self.action_to_id = {tuple(a): i for i, a in enumerate(self.actions)}

    def _build_actions(self) -> list[tuple[str, ...]]:
        required = tuple(sorted(self.required_sensor_ids, key=self.sensor_ids.index))
        optional = [sid for sid in self.sensor_ids if sid not in self.required_sensor_ids]
        actions: list[tuple[str, ...]] = [required]
        steady_limit = self._steady_limit()
        max_optional = max(self.max_active - len(required), 0)
        for k in range(1, max_optional + 1):
            for subset in itertools.combinations(optional, k):
                full_subset = tuple(sorted(required + tuple(subset), key=self.sensor_ids.index))
                if self.steady_power(full_subset) <= steady_limit + 1e-12:
                    actions.append(full_subset)
        deduped: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for action in actions:
            if action not in seen and self.steady_power(action) <= steady_limit + 1e-12:
                deduped.append(action)
                seen.add(action)
        return deduped

    def _steady_limit(self) -> float:
        return max(self.per_step_budget - self.safety_margin, 0.0)

    def _startup_limit(self) -> float | None:
        if self.startup_peak_budget is None:
            return None
        return max(self.startup_peak_budget - self.safety_margin, 0.0)

    def size(self) -> int:
        return len(self.actions)

    def decode(self, action_id: int) -> list[str]:
        return list(self.actions[action_id])

    def encode(self, subset: list[str] | tuple[str, ...]) -> int:
        key = tuple(sorted(subset, key=self.sensor_ids.index))
        if key in self.action_to_id:
            return self.action_to_id[key]
        return 0

    def steady_power(self, subset: Iterable[str]) -> float:
        return float(sum(self.power_costs[str(sid)] for sid in subset))

    def transition_peak_power(self, subset: Iterable[str], prev_selected: Iterable[str] | None = None) -> float:
        prev = set(prev_selected or [])
        total = 0.0
        for sid in subset:
            sid = str(sid)
            if sid in prev:
                total += self.power_costs[sid]
            else:
                total += self.startup_peak_costs[sid]
        return float(total)

    def power_metrics(self, subset: Iterable[str], prev_selected: Iterable[str] | None = None) -> dict[str, float]:
        subset_list = [str(sid) for sid in subset]
        prev = set(prev_selected or [])
        steady = self.steady_power(subset_list)
        peak = self.transition_peak_power(subset_list, prev)
        startup_extra = float(
            sum(
                max(self.startup_peak_costs[sid] - self.power_costs[sid], 0.0)
                for sid in subset_list
                if sid not in prev
            )
        )
        return {
            "steady_power": steady,
            "peak_power": peak,
            "startup_extra_power": startup_extra,
            "startup_count": float(sum(1 for sid in subset_list if sid not in prev)),
        }

    def is_subset_feasible(self, subset: Iterable[str], prev_selected: Iterable[str] | None = None) -> bool:
        subset_list = [str(sid) for sid in subset]
        if len(subset_list) > self.max_active:
            return False
        if any(req not in subset_list for req in self.required_sensor_ids):
            return False
        if self.steady_power(subset_list) > self._steady_limit() + 1e-12:
            return False
        startup_limit = self._startup_limit()
        if prev_selected is not None and startup_limit is not None:
            if self.transition_peak_power(subset_list, prev_selected) > startup_limit + 1e-12:
                return False
        return True

    def is_action_feasible(self, action_id: int, prev_selected: Iterable[str] | None = None) -> bool:
        return self.is_subset_feasible(self.actions[action_id], prev_selected)

    def feasible_action_ids(self, prev_selected: Iterable[str] | None = None) -> list[int]:
        return [i for i, subset in enumerate(self.actions) if self.is_subset_feasible(subset, prev_selected)]

    def feasible_action_mask(self, prev_selected: Iterable[str] | None = None) -> np.ndarray:
        mask = np.zeros(self.size(), dtype=bool)
        for aid in self.feasible_action_ids(prev_selected):
            mask[aid] = True
        return mask

    def nearest_feasible(self, ranked_sensor_ids: list[str], prev_selected: Iterable[str] | None = None) -> int:
        ranked = [str(sid) for sid in ranked_sensor_ids]
        ordered: list[str] = []
        for sid in self.required_sensor_ids + ranked:
            if sid not in ordered:
                ordered.append(sid)
        max_optional = max(self.max_active - len(self.required_sensor_ids), 0)
        optional_ranked = [sid for sid in ordered if sid not in self.required_sensor_ids]
        for k in range(min(len(optional_ranked), max_optional), -1, -1):
            subset = tuple(sorted(self.required_sensor_ids + optional_ranked[:k], key=self.sensor_ids.index))
            aid = self.action_to_id.get(subset)
            if aid is not None and self.is_action_feasible(aid, prev_selected):
                return aid
        feasible = self.feasible_action_ids(prev_selected)
        return int(feasible[0]) if feasible else 0

    def sanitize_action_id(self, action_id: int, prev_selected: Iterable[str] | None = None) -> int:
        if self.is_action_feasible(action_id, prev_selected):
            return int(action_id)
        return self.nearest_feasible(self.decode(action_id), prev_selected)
