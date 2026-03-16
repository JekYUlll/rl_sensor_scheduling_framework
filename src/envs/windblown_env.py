from __future__ import annotations

import math

import numpy as np

from envs.base_environment import BaseEnvironment
from sensors.base_sensor import SensorSpec
from sensors.windblown_sensor import WindblownSensor


class WindblownEnvironment(BaseEnvironment):
    """Business-case adapter environment (standalone, no legacy import)."""

    def __init__(self, env_cfg: dict, sensor_cfg: dict) -> None:
        self.cfg = env_cfg
        self._t = 0
        self._horizon = int(env_cfg.get("horizon", 2000))
        self._storm = 0
        self._latent = self._init_state()
        self.sensors = self._build_sensors(sensor_cfg)

    def _build_sensors(self, sensor_cfg: dict) -> dict[str, WindblownSensor]:
        sensors: dict[str, WindblownSensor] = {}
        for item in sensor_cfg.get("sensors", []):
            spec = SensorSpec(
                sensor_id=item["sensor_id"],
                obs_dim=len(item.get("variables", [])),
                variables=list(item.get("variables", [])),
                refresh_interval=int(item.get("refresh_interval", 1)),
                power_cost=float(item.get("power_cost", 1.0)),
                startup_delay=int(item.get("startup_delay", 0)),
                noise_std=item.get("noise_std", 0.0),
            )
            sensors[spec.sensor_id] = WindblownSensor(spec)
        return sensors

    def _init_state(self) -> dict[str, float]:
        return {
            "wind_speed_ms": 7.0,
            "wind_direction_deg": 240.0,
            "air_temperature_c": -14.0,
            "relative_humidity": 68.0,
            "air_pressure_pa": 68400.0,
            "snow_mass_flux_kg_m2_s": 1e-5,
            "snow_number_flux_m2_s": 2500.0,
        }

    def _step_latent(self) -> None:
        p01 = float(self.cfg.get("storm_transition", {}).get("p01", 0.02))
        p10 = float(self.cfg.get("storm_transition", {}).get("p10", 0.04))
        if self._storm == 0 and np.random.rand() < p01:
            self._storm = 1
        elif self._storm == 1 and np.random.rand() < p10:
            self._storm = 0

        t_day = (self._t % 86400) / 86400.0
        daily = math.sin(2.0 * math.pi * t_day)
        wind = self._latent["wind_speed_ms"] + 0.02 * (8.0 + 1.5 * daily - self._latent["wind_speed_ms"])
        wind += np.random.randn() * float(self.cfg.get("noise", {}).get("wind_speed", 0.4))
        if self._storm:
            wind += 1.5

        temp = self._latent["air_temperature_c"] + 0.02 * (-13.0 + 3.0 * daily - self._latent["air_temperature_c"])
        temp += np.random.randn() * float(self.cfg.get("noise", {}).get("temperature", 0.2))

        rh = self._latent["relative_humidity"] + 0.03 * (65.0 - self._latent["relative_humidity"])
        rh += np.random.randn() * float(self.cfg.get("noise", {}).get("humidity", 0.8))
        if self._storm:
            rh += 2.0
        rh = max(1.0, min(100.0, rh))

        pressure = self._latent["air_pressure_pa"] + np.random.randn() * 8.0
        direction = (self._latent["wind_direction_deg"] + np.random.randn() * 2.5) % 360.0

        threshold = max(6.5, 0.45 * abs(temp) ** 0.5 + 5.5)
        exceed = max(0.0, wind - threshold)
        mass_flux = max(0.0, 2e-6 + 3e-6 * exceed + (2e-5 if self._storm else 0.0) + np.random.randn() * 2e-6)
        number_flux = max(0.0, 400.0 + 2e5 * mass_flux + np.random.randn() * 120.0)

        self._latent.update(
            {
                "wind_speed_ms": float(wind),
                "wind_direction_deg": float(direction),
                "air_temperature_c": float(temp),
                "relative_humidity": float(rh),
                "air_pressure_pa": float(pressure),
                "snow_mass_flux_kg_m2_s": float(mass_flux),
                "snow_number_flux_m2_s": float(number_flux),
            }
        )

    def reset(self) -> dict:
        self._t = 0
        self._storm = 0
        self._latent = self._init_state()
        return {
            "latent_state": dict(self._latent),
            "available_observations": [],
            "event_flags": {"storm": False},
            "done": False,
            "info": {"t": self._t},
        }

    def step(self, action: list[str]) -> dict:
        self._t += 1
        self._step_latent()
        observations = []
        total_power = 0.0
        for sid in action:
            sensor = self.sensors.get(sid)
            if sensor is None:
                continue
            obs = sensor.observe(self._latent, t=self._t)
            if obs.get("available", False):
                observations.append(obs)
                total_power += float(obs.get("power_cost", 0.0))

        return {
            "latent_state": dict(self._latent),
            "available_observations": observations,
            "event_flags": {"storm": bool(self._storm)},
            "done": self._t >= self._horizon,
            "info": {"t": self._t, "power_cost": total_power},
        }

    def get_ground_truth(self) -> dict:
        return dict(self._latent)

    def get_time_index(self) -> int:
        return self._t
