from __future__ import annotations
import itertools
from collections.abc import Iterable
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SensorSubset:
    sensors: tuple[str, ...]
    score: float

class OnlineSubsetProjector:
    """Constraint-aware online subset selector.

    This class is intended for business cases where static action-id enumeration
    is too restrictive. For small sensor sets it performs exact feasible-subset
    search at runtime; for larger sets it falls back to a greedy approximation.
    """

    def __init__(self, sensor_ids: list[str], power_costs: dict[str, float], max_active: int, per_step_budget: float, startup_peak_costs: dict[str, float] | None=None, startup_peak_budget: float | None=None, safety_margin: float=0.0, exact_search_max_sensors: int=12) -> None:
        self.sensor_ids = [str(sid) for sid in sensor_ids]
        self.power_costs = {str(k): float(v) for k, v in power_costs.items()}
        self.startup_peak_costs = {sid: float((startup_peak_costs or {}).get(sid, self.power_costs[sid])) for sid in self.sensor_ids}
        self.max_active = int(max_active)
        self.per_step_budget = float(per_step_budget)
        self.startup_peak_budget = None if startup_peak_budget is None else float(startup_peak_budget)
        self.safety_margin = max(0.0, float(safety_margin))
        self.exact_search_max_sensors = int(exact_search_max_sensors)
        self._subset_cache = self._build_all_subsets()

    def _build_all_subsets(self) -> list[tuple[str, ...]]:
        subsets: list[tuple[str, ...]] = [tuple()]
        max_k = min(self.max_active, len(self.sensor_ids))
        for k in range(1, max_k + 1):
            for subset in itertools.combinations(self.sensor_ids, k):
                subsets.append(tuple(subset))
        return subsets

    def _steady_limit(self) -> float:
        return max(self.per_step_budget - self.safety_margin, 0.0)

    def _startup_limit(self) -> float | None:
        if self.startup_peak_budget is None:
            return None
        return max(self.startup_peak_budget - self.safety_margin, 0.0)

    def steady_power(self, subset: Iterable[str]) -> float:
        return float(sum((self.power_costs[str(sid)] for sid in subset)))

    def transition_peak_power(self, subset: Iterable[str], prev_selected: Iterable[str] | None=None) -> float:
        prev = set(prev_selected or [])
        total = 0.0
        for sid in subset:
            sid = str(sid)
            if sid in prev:
                total += self.power_costs[sid]
            else:
                total += self.startup_peak_costs[sid]
        return float(total)

    def power_metrics(self, subset: Iterable[str], prev_selected: Iterable[str] | None=None) -> dict[str, float]:
        subset_list = [str(sid) for sid in subset]
        prev = set(prev_selected or [])
        steady = self.steady_power(subset_list)
        peak = self.transition_peak_power(subset_list, prev)
        startup_extra = float(sum((max(self.startup_peak_costs[sid] - self.power_costs[sid], 0.0) for sid in subset_list if sid not in prev)))
        return {'steady_power': steady, 'peak_power': peak, 'startup_extra_power': startup_extra, 'startup_count': float(sum((1 for sid in subset_list if sid not in prev)))}

    def is_subset_feasible(self, subset: Iterable[str], prev_selected: Iterable[str] | None=None) -> bool:
        subset_list = [str(sid) for sid in subset]
        if len(subset_list) > self.max_active:
            return False
        if self.steady_power(subset_list) > self._steady_limit() + 1e-12:
            return False
        startup_limit = self._startup_limit()
        if startup_limit is not None:
            if self.transition_peak_power(subset_list, prev_selected) > startup_limit + 1e-12:
                return False
        return True

    def feasible_subsets(
        self,
        prev_selected: Iterable[str] | None = None,
        *,
        allow_empty: bool = True,
    ) -> list[tuple[str, ...]]:
        feasible = [subset for subset in self._subset_cache if self.is_subset_feasible(subset, prev_selected)]
        if allow_empty:
            return feasible
        non_empty = [subset for subset in feasible if len(subset) > 0]
        return non_empty if non_empty else feasible

    def size(self) -> int:
        return len(self._subset_cache)

    def sample_random_subset(self, prev_selected: Iterable[str] | None=None, rng: np.random.Generator | None=None, *, allow_empty: bool=False) -> list[str]:
        feasible = self.feasible_subsets(prev_selected, allow_empty=allow_empty)
        if not feasible:
            return []
        generator = rng or np.random.default_rng()
        idx = int(generator.integers(0, len(feasible)))
        return list(feasible[idx])

    def _subset_score(self, subset: Iterable[str], score_map: dict[str, float]) -> float:
        return float(sum((score_map.get(str(sid), 0.0) for sid in subset)))

    def select_from_scores(self, score_map: dict[str, float], prev_selected: Iterable[str] | None=None) -> list[str]:
        ordered_scores = {str(k): float(v) for k, v in score_map.items()}
        if len(self.sensor_ids) <= self.exact_search_max_sensors:
            best = SensorSubset(tuple(), float('-inf'))
            for subset in self.feasible_subsets(prev_selected, allow_empty=False):
                score = self._subset_score(subset, ordered_scores)
                if score > best.score + 1e-12 or (abs(score - best.score) <= 1e-12 and len(subset) > len(best.sensors)):
                    best = SensorSubset(tuple(subset), score)
            return list(best.sensors)
        ranked = [sid for sid, _ in sorted(ordered_scores.items(), key=lambda x: x[1], reverse=True)]
        return self.project_ranked(ranked, prev_selected=prev_selected)

    def project_ranked(self, ranked_sensor_ids: list[str], prev_selected: Iterable[str] | None=None) -> list[str]:
        ordered: list[str] = []
        for sid in ranked_sensor_ids:
            sid = str(sid)
            if sid in self.sensor_ids and sid not in ordered:
                ordered.append(sid)
        if len(self.sensor_ids) <= self.exact_search_max_sensors:
            score_map = {sid: float(len(ordered) - idx) for idx, sid in enumerate(ordered)}
            return self.select_from_scores(score_map, prev_selected=prev_selected)
        chosen: list[str] = []
        for sid in ordered:
            candidate = chosen + [sid]
            if self.is_subset_feasible(candidate, prev_selected):
                chosen = candidate
        return chosen
