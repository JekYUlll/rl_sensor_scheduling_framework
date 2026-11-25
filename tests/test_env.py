from __future__ import annotations

from core.config import load_yaml
from envs.linear_gaussian_env import LinearGaussianEnvironment


def test_linear_env_step_runs():
    env_cfg = load_yaml("configs/env/linear_gaussian_case.yaml")
    sensor_cfg = load_yaml("configs/sensors/linear_gaussian_sensors.yaml")
    env = LinearGaussianEnvironment.from_dict(env_cfg, sensor_cfg)
    env.reset()
    out = env.step(["s0", "s1"])
    assert "latent_state" in out
    assert "available_observations" in out
    assert "event_flags" in out
