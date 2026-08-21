#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402


DEFAULT_STATE_COLUMNS = (
    "wind_speed_ms",
    "wind_direction_deg",
    "wind_dir_sin",
    "wind_dir_cos",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
    "solar_radiation_wm2",
    "snow_surface_temperature_c",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
    "snow_mass_flux_kg_m2_s",
)

DEFAULT_REQUIRED_SENSOR_IDS: tuple[str, ...] = ()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal v2 random projected-score rollout.")
    parser.add_argument("--truth-csv", default="data/generated/public_weather_truth_smoke.csv")
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced.yaml")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-step-budget", type=float, default=2.3)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--required-sensors", nargs="*", default=list(DEFAULT_REQUIRED_SENSOR_IDS))
    args = parser.parse_args()

    truth = pd.read_csv(args.truth_csv)
    sensors = load_sensor_specs(args.sensor_cfg)
    constraints = PowerConstraintsV2(
        max_active=int(args.max_active),
        per_step_budget=float(args.per_step_budget),
        startup_peak_budget=float(args.startup_peak_budget),
        required_sensor_ids=tuple(str(sensor_id) for sensor_id in args.required_sensors),
    )
    env = WarmupSchedulingEnv(
        truth,
        sensors,
        constraints,
        WarmupEnvConfig(state_columns=DEFAULT_STATE_COLUMNS, lookback=20, episode_len=int(args.steps), seed=int(args.seed)),
    )
    rng = np.random.default_rng(int(args.seed))
    state, info = env.reset()
    del state, info
    rewards = []
    powers = []
    peaks = []
    aborts = []
    for _ in range(int(args.steps)):
        scores = rng.normal(size=len(sensors))
        _, reward, done, info = env.step_scores(scores)
        rewards.append(float(reward))
        powers.append(float(info["power"]))
        peaks.append(float(info["peak_power"]))
        aborts.append(int(info["warmup_abort_count"]))
        if done:
            break
    print(
        {
            "steps": len(rewards),
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "power_mean": float(np.mean(powers)) if powers else 0.0,
            "peak_max": float(np.max(peaks)) if peaks else 0.0,
            "warmup_aborts": int(aborts[-1]) if aborts else 0,
        }
    )


if __name__ == "__main__":
    main()
