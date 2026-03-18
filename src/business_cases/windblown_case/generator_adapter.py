from __future__ import annotations

from pathlib import Path

import pandas as pd

from business_cases.windblown_case.adapter import WindblownCaseAdapter


def generate_windblown_csv(env_cfg_path: str, sensor_cfg_path: str, n_steps: int, out_csv: str) -> str:
    adapter = WindblownCaseAdapter(env_cfg_path, sensor_cfg_path)
    env = adapter.build_environment()
    env_cfg = adapter.env_cfg
    base_freq_s = int(env_cfg.get("base_freq_s", 1))
    start_ts = pd.Timestamp(str(env_cfg.get("start_timestamp", "2026-01-01 00:00:00")))
    env.reset()

    rows = []
    all_sensors = adapter.sensor_ids()
    for step_idx in range(int(n_steps)):
        step = env.step(all_sensors)
        latent = dict(step["latent_state"])
        latent["time_idx"] = env.get_time_index()
        latent["timestamp"] = start_ts + pd.Timedelta(seconds=step_idx * base_freq_s)
        latent["storm_flag"] = bool(step.get("event_flags", {}).get("storm", False))
        latent["event_flag"] = bool(step.get("event_flags", {}).get("storm", False))
        rows.append(latent)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)
