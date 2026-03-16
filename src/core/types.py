from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    latent_state: Any
    available_observations: list[dict]
    event_flags: dict[str, Any]
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: str
    metrics_path: str
    config_path: str
