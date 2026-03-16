from __future__ import annotations

from pathlib import Path

import pandas as pd

from business_cases.windblown_case.adapter import WindblownCaseAdapter


def generate_windblown_csv(env_cfg_path: str, sensor_cfg_path: str, n_steps: int, out_csv: str) -> str:
    adapter = WindblownCaseAdapter(env_cfg_path, sensor_cfg_path)
    env = adapter.build_environment()
    env.reset()

    rows = []
    all_sensors = adapter.sensor_ids()
    for _ in range(int(n_steps)):
        step = env.step(all_sensors)
        latent = step["latent_state"]
        latent["t"] = env.get_time_index()
        rows.append(latent)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)
