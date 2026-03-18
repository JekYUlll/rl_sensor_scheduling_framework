from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from envs.base_environment import BaseEnvironment
from sensors.dataset_sensor import DatasetSensor


@dataclass
class TruthReplayConfig:
    state_columns: list[str]
    split_start: int
    split_end: int
    episode_len: int
    random_reset: bool
    base_freq_s: int = 1
    event_column: str = "event_flag"


class TruthReplayEnvironment(BaseEnvironment):
    def __init__(
        self,
        truth_df: pd.DataFrame,
        sensors: list[DatasetSensor],
        cfg: TruthReplayConfig,
        seed: int = 42,
    ) -> None:
        self.truth_df = truth_df.reset_index(drop=True)
        self.state_columns = list(cfg.state_columns)
        self.cfg = cfg
        self.sensors = {sensor.sensor_id: sensor for sensor in sensors}
        self.rng = np.random.default_rng(seed)
        self._current_idx = cfg.split_start
        self._episode_start_idx = cfg.split_start
        self._episode_end_idx = min(cfg.split_end, cfg.split_start + cfg.episode_len)
        self._local_t = 0
        self._values = self.truth_df[self.state_columns].to_numpy(dtype=float)
        self._event_flags = self.truth_df[cfg.event_column].astype(bool).to_numpy() if cfg.event_column in self.truth_df.columns else np.zeros(len(self.truth_df), dtype=bool)

    def _reset_sensors(self) -> None:
        for sensor in self.sensors.values():
            sensor.reset()

    def _choose_start_idx(self) -> int:
        max_start = max(self.cfg.split_start, self.cfg.split_end - self.cfg.episode_len)
        if (not self.cfg.random_reset) or max_start <= self.cfg.split_start:
            return self.cfg.split_start
        return int(self.rng.integers(self.cfg.split_start, max_start + 1))

    def _row_to_state(self, idx: int) -> dict[str, float]:
        row = self._values[idx]
        return {name: float(row[i]) for i, name in enumerate(self.state_columns)}

    def reset(self) -> dict:
        self._episode_start_idx = self._choose_start_idx()
        self._episode_end_idx = min(self.cfg.split_end, self._episode_start_idx + self.cfg.episode_len)
        self._current_idx = self._episode_start_idx
        self._local_t = 0
        self._reset_sensors()
        return {
            "latent_state": self._row_to_state(self._current_idx),
            "available_observations": [],
            "event_flags": {"event": bool(self._event_flags[self._current_idx])},
            "done": False,
            "info": {"t": self._local_t, "row_idx": self._current_idx},
        }

    def step(self, action: list[str]) -> dict:
        next_idx = min(self._current_idx + 1, self.cfg.split_end - 1)
        self._current_idx = next_idx
        self._local_t += 1
        latent_state = self._row_to_state(self._current_idx)
        observations = []
        total_power = 0.0
        for sensor_id in action:
            sensor = self.sensors.get(sensor_id)
            if sensor is None:
                continue
            obs = sensor.observe(latent_state, t=self._local_t)
            if obs.get("available", False):
                observations.append(obs)
                total_power += float(obs.get("power_cost", 0.0))

        done = self._current_idx >= (self._episode_end_idx - 1)
        return {
            "latent_state": latent_state,
            "available_observations": observations,
            "event_flags": {"event": bool(self._event_flags[self._current_idx])},
            "done": done,
            "info": {
                "t": self._local_t,
                "row_idx": self._current_idx,
                "power_cost": total_power,
            },
        }

    def get_ground_truth(self) -> dict:
        return self._row_to_state(self._current_idx)

    def get_time_index(self) -> int:
        return self._local_t
