from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self) -> dict:
        """Reset environment and return initial latent state / info."""

    @abstractmethod
    def step(self, action: Any) -> dict:
        """Advance one step and return latent/obs/event payload."""

    @abstractmethod
    def get_ground_truth(self) -> dict:
        """Return current latent truth."""

    @abstractmethod
    def get_time_index(self) -> int:
        """Return current discrete time step."""


class BaseSensor(ABC):
    @property
    @abstractmethod
    def sensor_id(self) -> str:
        """Unique sensor id."""

    @abstractmethod
    def power_cost(self) -> float:
        """Current activation cost."""

    @abstractmethod
    def can_sample(self, t: int) -> bool:
        """Whether sensor is available at time t."""

    @abstractmethod
    def observe(self, latent_state: Any) -> dict:
        """Observe latent state."""


class BaseEstimator(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def predict(self) -> None:
        ...

    @abstractmethod
    def update(self, observations: list[dict]) -> None:
        ...

    @abstractmethod
    def get_state_estimate(self) -> Any:
        ...

    @abstractmethod
    def get_uncertainty_summary(self) -> dict:
        ...

    @abstractmethod
    def get_rl_state_features(self) -> dict:
        ...


class BaseScheduler(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def act(self, state: dict) -> Any:
        ...

    def observe_transition(self, transition: dict) -> None:
        """Optional hook for RL schedulers."""


class BasePredictor(ABC):
    @abstractmethod
    def fit(self, train_data, val_data=None) -> None:
        ...

    @abstractmethod
    def predict(self, test_data) -> np.ndarray:
        ...

    @abstractmethod
    def evaluate(self, test_data) -> dict:
        ...

    def set_context(
        self,
        input_feature_names: list[str],
        target_feature_names: list[str],
        stats: dict[str, np.ndarray],
    ) -> None:
        """Optional hook for predictors that need dataset metadata or scaling stats."""
