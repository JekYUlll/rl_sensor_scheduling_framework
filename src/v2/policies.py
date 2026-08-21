from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class V2Policy:
    name: str

    def reset(self) -> None:
        pass

    def act_scores(self, env: object) -> np.ndarray:
        raise NotImplementedError


@dataclass
class RandomScorePolicy(V2Policy):
    n_sensors: int
    seed: int = 42
    name: str = "random"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(int(self.seed))

    def reset(self) -> None:
        self.rng = np.random.default_rng(int(self.seed))

    def act_scores(self, env: object) -> np.ndarray:
        del env
        return self.rng.normal(size=int(self.n_sensors))


@dataclass
class FeasibleStaticPriorityPolicy(V2Policy):
    n_sensors: int
    name: str = "feasible_static_projected"

    def act_scores(self, env: object) -> np.ndarray:
        del env
        return np.linspace(1.0, 0.0, int(self.n_sensors))


@dataclass
class StaticMaskPolicy(V2Policy):
    mask: tuple[bool, ...]
    name: str = "oracle_static_projected"

    def act_mask(self, env: object) -> np.ndarray:
        del env
        return np.asarray(self.mask, dtype=bool)

    def act_scores(self, env: object) -> np.ndarray:
        del env
        mask = np.asarray(self.mask, dtype=bool)
        return np.where(mask, 1.0, -1.0)


@dataclass
class MinDwellPolicyWrapper(V2Policy):
    base_policy: V2Policy
    min_dwell_steps: int = 1
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"dwell{max(1, int(self.min_dwell_steps))}_{self.base_policy.name}"
        self._hold_remaining = 0
        self._held_mask: np.ndarray | None = None

    def reset(self) -> None:
        self.base_policy.reset()
        self._hold_remaining = 0
        self._held_mask = None

    def act_mask(self, env: object) -> np.ndarray:
        if self._hold_remaining > 0 and self._held_mask is not None:
            self._hold_remaining -= 1
            return np.asarray(self._held_mask, dtype=bool).copy()

        desired = self._project_desired_mask(env)
        prev = np.asarray(getattr(env, "previous_action_mask"), dtype=bool).reshape(-1)
        dwell = max(1, int(self.min_dwell_steps))
        if dwell > 1 and desired.shape == prev.shape and not np.array_equal(desired, prev):
            self._held_mask = np.asarray(desired, dtype=bool).copy()
            self._hold_remaining = dwell - 1
        else:
            self._held_mask = np.asarray(desired, dtype=bool).copy()
            self._hold_remaining = 0
        return np.asarray(desired, dtype=bool)

    def act_scores(self, env: object) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _project_desired_mask(self, env: object) -> np.ndarray:
        act_mask = getattr(self.base_policy, "act_mask", None)
        projector = getattr(env, "projector")
        runtimes = getattr(env, "runtimes")
        if callable(act_mask):
            desired = np.asarray(act_mask(env), dtype=bool).reshape(-1)
            return np.asarray(projector.project_mask(desired, runtimes).selected_mask, dtype=bool)

        scores = np.asarray(self.base_policy.act_scores(env), dtype=float).reshape(-1)
        duty_adjusted = getattr(env, "_duty_adjusted_scores", None)
        if callable(duty_adjusted):
            scores = np.asarray(duty_adjusted(scores), dtype=float).reshape(-1)
        return np.asarray(projector.project_scores(scores, runtimes).selected_mask, dtype=bool)


@dataclass
class FullOpenUnconstrainedScorePolicy(V2Policy):
    n_sensors: int
    name: str = "full_open_unconstrained"

    def act_scores(self, env: object) -> np.ndarray:
        del env
        return np.ones(int(self.n_sensors), dtype=float)


@dataclass
class RoundRobinScorePolicy(V2Policy):
    n_sensors: int
    group_size: int = 2
    name: str = "round_robin"

    def reset(self) -> None:
        self.t = 0

    def __post_init__(self) -> None:
        self.t = 0

    def act_scores(self, env: object) -> np.ndarray:
        del env
        scores = np.full(int(self.n_sensors), -1.0, dtype=float)
        for offset in range(max(1, int(self.group_size))):
            scores[(self.t + offset) % int(self.n_sensors)] = 1.0 - 0.01 * offset
        self.t += 1
        return scores


@dataclass
class AoIScorePolicy(V2Policy):
    n_sensors: int
    name: str = "aoi"

    def act_scores(self, env: object) -> np.ndarray:
        runtimes = getattr(env, "runtimes")
        sensor_ids = getattr(env, "sensor_ids")
        current_idx = int(getattr(env, "current_idx"))
        return np.asarray([runtimes[sid].freshness(current_idx) for sid in sensor_ids], dtype=float)


def default_policies(n_sensors: int, *, seed: int = 42) -> list[V2Policy]:
    return [
        FeasibleStaticPriorityPolicy(n_sensors=n_sensors),
        RoundRobinScorePolicy(n_sensors=n_sensors, group_size=max(1, n_sensors // 3)),
        AoIScorePolicy(n_sensors=n_sensors),
        RandomScorePolicy(n_sensors=n_sensors, seed=seed),
    ]
