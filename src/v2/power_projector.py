from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v2.sensor_spec import SensorSpecV2
from v2.warmup_state import SensorMode, SensorRuntime


@dataclass(frozen=True)
class PowerConstraintsV2:
    max_active: int | None = None
    per_step_budget: float | None = None
    startup_peak_budget: float | None = None
    required_sensor_ids: tuple[str, ...] = ()
    coverage_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class ProjectionResult:
    selected_sensor_ids: tuple[str, ...]
    selected_mask: np.ndarray
    steady_power: float
    peak_power: float
    feasible: bool


class PowerProjector:
    def __init__(self, sensor_specs: list[SensorSpecV2], constraints: PowerConstraintsV2) -> None:
        self.sensor_specs = list(sensor_specs)
        self.constraints = constraints
        self.sensor_ids = tuple(spec.sensor_id for spec in self.sensor_specs)
        self.required_sensor_ids = tuple(str(sensor_id) for sensor_id in constraints.required_sensor_ids)
        unknown_required = [sensor_id for sensor_id in self.required_sensor_ids if sensor_id not in self.sensor_ids]
        if unknown_required:
            raise ValueError(f"required_sensor_ids are not defined sensors: {unknown_required}")
        self.required_indices = tuple(self.sensor_ids.index(sensor_id) for sensor_id in self.required_sensor_ids)
        self.coverage_groups = self._normalise_coverage_groups(constraints.coverage_groups)

    def project_scores(self, scores: np.ndarray, runtimes: dict[str, SensorRuntime]) -> ProjectionResult:
        scores_arr = np.asarray(scores, dtype=float).reshape(-1)
        if scores_arr.shape[0] != len(self.sensor_specs):
            raise ValueError(f"scores length {scores_arr.shape[0]} != number of sensors {len(self.sensor_specs)}")
        order = np.argsort(-scores_arr)
        selected = self._required_selection(runtimes)
        selected = self._coverage_selection(selected, scores_arr, runtimes)
        for idx in order:
            if int(idx) in selected:
                continue
            candidate = selected + [int(idx)]
            if self._is_feasible(candidate, runtimes):
                selected = candidate
        return self._result(selected, runtimes)

    def project_mask(self, desired_mask: np.ndarray, runtimes: dict[str, SensorRuntime]) -> ProjectionResult:
        mask = np.asarray(desired_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != len(self.sensor_specs):
            raise ValueError(f"mask length {mask.shape[0]} != number of sensors {len(self.sensor_specs)}")
        selected = self._required_selection(runtimes)
        scores_arr = mask.astype(float)
        selected = self._coverage_selection(selected, scores_arr, runtimes)
        for idx in np.flatnonzero(mask):
            if int(idx) in selected:
                continue
            candidate = selected + [int(idx)]
            if self._is_feasible(candidate, runtimes):
                selected = candidate
        return self._result(selected, runtimes)

    def _required_selection(self, runtimes: dict[str, SensorRuntime]) -> list[int]:
        selected: list[int] = []
        for idx in self.required_indices:
            candidate = selected + [int(idx)]
            if not self._is_feasible(candidate, runtimes):
                sensor_id = self.sensor_specs[int(idx)].sensor_id
                raise ValueError(
                    f"Required sensor '{sensor_id}' cannot satisfy current power constraints. "
                    "Relax constraints or remove it from required_sensor_ids."
                )
            selected = candidate
        return selected

    def _coverage_selection(
        self,
        selected: list[int],
        scores: np.ndarray,
        runtimes: dict[str, SensorRuntime],
    ) -> list[int]:
        uncovered: list[tuple[str, tuple[int, ...]]] = []
        for group_name, group_indices in self.coverage_groups:
            if any(idx in selected for idx in group_indices):
                continue
            candidates = tuple(idx for idx in group_indices if idx not in selected)
            if not candidates:
                raise ValueError(
                    f"Coverage group '{group_name}' cannot satisfy current power constraints. "
                    "Relax constraints or reduce coverage_groups."
                )
            uncovered.append((group_name, candidates))
        if not uncovered:
            return selected

        best_selection: list[int] | None = None
        best_score = -float("inf")

        def search(group_idx: int, current: list[int], score_sum: float) -> None:
            nonlocal best_score, best_selection
            if group_idx >= len(uncovered):
                if self._is_feasible(current, runtimes) and score_sum > best_score:
                    best_score = float(score_sum)
                    best_selection = list(current)
                return
            _, group_indices = uncovered[group_idx]
            ordered = sorted(group_indices, key=lambda idx: float(scores[idx]), reverse=True)
            for idx in ordered:
                candidate = current + [int(idx)]
                if self._is_feasible(candidate, runtimes):
                    search(group_idx + 1, candidate, score_sum + float(scores[idx]))

        search(0, list(selected), 0.0)
        if best_selection is None:
            names = ", ".join(name for name, _ in uncovered)
            raise ValueError(
                f"Coverage groups cannot jointly satisfy current power constraints: {names}. "
                "Relax constraints or reduce coverage_groups."
            )
        return best_selection

    def _is_feasible(self, indices: list[int], runtimes: dict[str, SensorRuntime]) -> bool:
        if self.constraints.max_active is not None and len(indices) > int(self.constraints.max_active):
            return False
        steady, peak = self._power(indices, runtimes)
        if self.constraints.per_step_budget is not None and steady > float(self.constraints.per_step_budget) + 1e-12:
            return False
        if self.constraints.startup_peak_budget is not None and peak > float(self.constraints.startup_peak_budget) + 1e-12:
            return False
        return True

    def _power(self, indices: list[int], runtimes: dict[str, SensorRuntime]) -> tuple[float, float]:
        steady = 0.0
        peak = 0.0
        for idx in indices:
            spec = self.sensor_specs[idx]
            runtime = runtimes[spec.sensor_id]
            steady += float(spec.power_cost)
            if runtime.mode == SensorMode.OFF:
                peak += float(max(spec.power_cost, spec.startup_peak_power))
            else:
                peak += float(spec.power_cost)
        return steady, peak

    def _result(self, indices: list[int], runtimes: dict[str, SensorRuntime]) -> ProjectionResult:
        mask = np.zeros(len(self.sensor_specs), dtype=bool)
        mask[indices] = True
        steady, peak = self._power(indices, runtimes)
        ids = tuple(self.sensor_specs[idx].sensor_id for idx in indices)
        return ProjectionResult(ids, mask, steady, peak, self._is_feasible(indices, runtimes))

    def _normalise_coverage_groups(
        self,
        coverage_groups: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        groups: list[tuple[str, tuple[int, ...]]] = []
        for raw_name, raw_sensor_ids in coverage_groups:
            group_name = str(raw_name)
            sensor_ids = tuple(str(sensor_id) for sensor_id in raw_sensor_ids)
            if not sensor_ids:
                continue
            unknown = [sensor_id for sensor_id in sensor_ids if sensor_id not in self.sensor_ids]
            if unknown:
                raise ValueError(f"coverage group '{group_name}' contains undefined sensors: {unknown}")
            groups.append((group_name, tuple(self.sensor_ids.index(sensor_id) for sensor_id in sensor_ids)))
        return tuple(groups)
