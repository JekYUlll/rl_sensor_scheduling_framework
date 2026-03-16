from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import StepResult
from envs.base_environment import BaseEnvironment
from sensors.base_sensor import SensorSpec
from sensors.linear_sensor import LinearSensor


@dataclass
class LinearEnvConfig:
    A: np.ndarray
    Q: np.ndarray
    x0: np.ndarray
    horizon: int
    event_norm_threshold: float = 2.5


class LinearGaussianEnvironment(BaseEnvironment):
    def __init__(self, cfg: LinearEnvConfig, sensors: list[LinearSensor]) -> None:
        self.cfg = cfg
        self.sensors = {s.sensor_id: s for s in sensors}
        self._t = 0
        self._x = cfg.x0.copy()

    @classmethod
    def from_dict(cls, env_cfg: dict, sensor_cfg: dict) -> "LinearGaussianEnvironment":
        A = np.asarray(env_cfg["A"], dtype=float)
        q_diag = np.asarray(env_cfg["Q_diag"], dtype=float)
        x0 = np.asarray(env_cfg["x0"], dtype=float)
        horizon = int(env_cfg.get("horizon", 400))
        th = float(env_cfg.get("event_threshold", {}).get("norm_l2", 2.5))
        cfg = LinearEnvConfig(A=A, Q=np.diag(q_diag), x0=x0, horizon=horizon, event_norm_threshold=th)
        sensors: list[LinearSensor] = []
        for item in sensor_cfg.get("sensors", []):
            c_vec = np.asarray(item["C"], dtype=float)
            r_diag = np.asarray(item.get("R_diag", [0.05]), dtype=float)
            spec = SensorSpec(
                sensor_id=item["sensor_id"],
                obs_dim=r_diag.shape[0],
                variables=item.get("variables", []),
                refresh_interval=int(item.get("refresh_interval", 1)),
                power_cost=float(item.get("power_cost", 1.0)),
                startup_delay=int(item.get("startup_delay", 0)),
                noise_std=float(np.sqrt(float(r_diag[0]))) if r_diag.size == 1 else list(np.sqrt(r_diag)),
            )
            sensors.append(LinearSensor(spec=spec, c_vec=c_vec, r_diag=r_diag))
        return cls(cfg=cfg, sensors=sensors)

    @property
    def state_dim(self) -> int:
        return int(self.cfg.A.shape[0])

    def reset(self) -> dict:
        self._t = 0
        self._x = self.cfg.x0.copy()
        return {
            "latent_state": self._x.copy(),
            "available_observations": [],
            "event_flags": {"event": bool(np.linalg.norm(self._x) > self.cfg.event_norm_threshold)},
            "done": False,
            "info": {"t": self._t},
        }

    def step(self, action: list[str]) -> dict:
        w = np.random.multivariate_normal(np.zeros(self.state_dim), self.cfg.Q)
        self._x = self.cfg.A @ self._x + w
        self._t += 1

        observations = []
        total_power = 0.0
        for sid in action:
            sensor = self.sensors.get(sid)
            if sensor is None:
                continue
            obs = sensor.observe(self._x, t=self._t)
            if obs.get("available", False):
                observations.append(obs)
                total_power += float(obs.get("power_cost", 0.0))

        done = self._t >= self.cfg.horizon
        event = bool(np.linalg.norm(self._x) > self.cfg.event_norm_threshold)
        result = StepResult(
            latent_state=self._x.copy(),
            available_observations=observations,
            event_flags={"event": event},
            done=done,
            info={"t": self._t, "power_cost": total_power},
        )
        return result.__dict__

    def get_ground_truth(self) -> dict:
        return {"x": self._x.copy()}

    def get_time_index(self) -> int:
        return self._t
