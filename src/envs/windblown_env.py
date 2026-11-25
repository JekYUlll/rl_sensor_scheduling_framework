from __future__ import annotations

import math

import numpy as np

from envs.base_environment import BaseEnvironment
from sensors.base_sensor import SensorSpec
from sensors.windblown_sensor import WindblownSensor


class WindblownEnvironment(BaseEnvironment):
    """Standalone synthetic windblown-snow business environment."""

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
                warmup_steps=int(item.get("warmup_steps", 0)),
                startup_peak_power=None if item.get("startup_peak_power") is None else float(item.get("startup_peak_power")),
                warmup_observation_mode=str(item.get("warmup_observation_mode", "none")),
                warmup_noise_scale=float(item.get("warmup_noise_scale", 1.0)),
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
            "solar_radiation_wm2": 0.0,
            "snow_surface_temperature_c": -15.0,
            "snow_particle_mean_diameter_mm": 0.28,
            "snow_particle_mean_velocity_ms": 0.8,
            "snow_mass_flux_kg_m2_s": 1.0e-5,
        }

    def _step_latent(self) -> None:
        p01 = float(self.cfg.get("storm_transition", {}).get("p01", 0.02))
        p10 = float(self.cfg.get("storm_transition", {}).get("p10", 0.04))
        if self._storm == 0 and np.random.rand() < p01:
            self._storm = 1
        elif self._storm == 1 and np.random.rand() < p10:
            self._storm = 0

        t_day = (self._t % 86400) / 86400.0
        daily = math.sin(2.0 * math.pi * (t_day - 0.25))
        daylight = max(0.0, daily)

        wind = self._latent["wind_speed_ms"] + 0.02 * (8.2 + 1.2 * daily - self._latent["wind_speed_ms"])
        wind += np.random.randn() * float(self.cfg.get("noise", {}).get("wind_speed", 0.4))
        if self._storm:
            wind += 1.6
        wind = max(0.1, wind)

        temp = self._latent["air_temperature_c"] + 0.02 * (-13.0 + 3.5 * daily - self._latent["air_temperature_c"])
        temp += np.random.randn() * float(self.cfg.get("noise", {}).get("temperature", 0.2))

        rh_target = 66.0 - 8.0 * daylight + (5.0 if self._storm else 0.0)
        rh = self._latent["relative_humidity"] + 0.03 * (rh_target - self._latent["relative_humidity"])
        rh += np.random.randn() * float(self.cfg.get("noise", {}).get("humidity", 0.8))
        rh = max(1.0, min(100.0, rh))

        pressure = self._latent["air_pressure_pa"] + 0.02 * (68420.0 - self._latent["air_pressure_pa"])
        pressure += np.random.randn() * 8.0
        if self._storm:
            pressure -= 6.0

        direction = (self._latent["wind_direction_deg"] + np.random.randn() * 2.5) % 360.0

        radiation = 850.0 * daylight * (0.65 if self._storm else 1.0)
        radiation += np.random.randn() * float(self.cfg.get("noise", {}).get("radiation", 18.0))
        radiation = max(0.0, radiation)

        surface_temp_target = temp - 0.8 + 0.002 * radiation
        surface_temp = self._latent["snow_surface_temperature_c"] + 0.08 * (surface_temp_target - self._latent["snow_surface_temperature_c"])
        surface_temp += np.random.randn() * float(self.cfg.get("noise", {}).get("surface_temperature", 0.15))

        threshold = max(6.5, 0.45 * abs(temp) ** 0.5 + 5.5)
        exceed = max(0.0, wind - threshold)
        storm_boost = 1.5e-5 if self._storm else 0.0
        mass_flux = 1.0e-6 + 4.5e-6 * exceed + storm_boost
        mass_flux += np.random.randn() * float(self.cfg.get("noise", {}).get("flux", 2.0e-6))
        mass_flux = max(0.0, mass_flux)

        diameter = 0.25 + 0.045 * exceed + (0.05 if self._storm else 0.0)
        diameter += np.random.randn() * float(self.cfg.get("noise", {}).get("diameter", 0.03))
        diameter = min(5.0, max(0.2, diameter))

        # Particle velocity follows the blowing-snow transport regime but is not
        # identical to bulk wind speed; near-surface particles are typically slower.
        particle_velocity_target = 0.25 * wind + 1.4 * exceed + (0.6 if self._storm else 0.0)
        particle_velocity = self._latent["snow_particle_mean_velocity_ms"] + 0.15 * (
            particle_velocity_target - self._latent["snow_particle_mean_velocity_ms"]
        )
        particle_velocity += np.random.randn() * float(self.cfg.get("noise", {}).get("particle_velocity", 0.08))
        particle_velocity = min(20.0, max(0.2, particle_velocity))

        self._latent.update(
            {
                "wind_speed_ms": float(wind),
                "wind_direction_deg": float(direction),
                "air_temperature_c": float(temp),
                "relative_humidity": float(rh),
                "air_pressure_pa": float(pressure),
                "solar_radiation_wm2": float(radiation),
                "snow_surface_temperature_c": float(surface_temp),
                "snow_particle_mean_diameter_mm": float(diameter),
                "snow_particle_mean_velocity_ms": float(particle_velocity),
                "snow_mass_flux_kg_m2_s": float(mass_flux),
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
        selected_set = {str(sid) for sid in action}
        sensor_status: dict[str, dict[str, float | int | bool | str]] = {}
        powered_sensor_ids: list[str] = []
        warming_sensor_ids: list[str] = []
        ready_sensor_ids: list[str] = []
        for sid, sensor in self.sensors.items():
            status = sensor.begin_step(sid in selected_set, t=self._t)
            sensor_status[sid] = status
            if bool(status.get("powered", False)):
                powered_sensor_ids.append(sid)
                total_power += float(status.get("power_cost", 0.0))
            if bool(status.get("warming", False)):
                warming_sensor_ids.append(sid)
            if bool(status.get("ready", False)):
                ready_sensor_ids.append(sid)
        for sid in action:
            sensor = self.sensors.get(sid)
            if sensor is None:
                continue
            obs = sensor.observe(self._latent, t=self._t)
            if obs.get("available", False):
                observations.append(obs)

        return {
            "latent_state": dict(self._latent),
            "available_observations": observations,
            "sensor_status": sensor_status,
            "powered_sensor_ids": powered_sensor_ids,
            "warming_sensor_ids": warming_sensor_ids,
            "ready_sensor_ids": ready_sensor_ids,
            "event_flags": {"storm": bool(self._storm)},
            "done": self._t >= self._horizon,
            "info": {"t": self._t, "power_cost": total_power},
        }

    def get_ground_truth(self) -> dict:
        return dict(self._latent)

    def get_time_index(self) -> int:
        return self._t
