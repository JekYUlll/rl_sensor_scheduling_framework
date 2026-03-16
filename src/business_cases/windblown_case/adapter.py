from __future__ import annotations

from core.config import load_yaml
from envs.windblown_env import WindblownEnvironment


class WindblownCaseAdapter:
    def __init__(self, env_cfg_path: str, sensor_cfg_path: str) -> None:
        self.env_cfg = load_yaml(env_cfg_path)
        self.sensor_cfg = load_yaml(sensor_cfg_path)

    def build_environment(self) -> WindblownEnvironment:
        return WindblownEnvironment(self.env_cfg, self.sensor_cfg)

    def sensor_ids(self) -> list[str]:
        return [s["sensor_id"] for s in self.sensor_cfg.get("sensors", [])]
