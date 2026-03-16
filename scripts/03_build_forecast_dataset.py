#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import load_yaml
from pipelines.common import build_linear_stack
from scheduling.action_space import DiscreteActionSpace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", default="configs/env/linear_gaussian_case.yaml")
    parser.add_argument("--sensor_cfg", default="configs/sensors/linear_gaussian_sensors.yaml")
    parser.add_argument("--estimator_cfg", default="configs/estimator/kalman.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--out_npz", default="data/processed/forecast_dataset.npz")
    args = parser.parse_args()

    env, estimator, action_space, base_cfg, sensor_cfg = build_linear_stack(args.env_cfg, args.sensor_cfg, args.estimator_cfg)
    env.reset()
    estimator.reset()

    xs = []
    for _ in range(args.steps):
        aid = np.random.randint(0, action_space.size())
        selected = action_space.decode(int(aid))
        step = env.step(selected)
        estimator.predict()
        estimator.update(step["available_observations"])
        estimator.on_step(selected, power_ratio=step.get("info", {}).get("power_cost", 0.0) / max(action_space.per_step_budget, 1e-6))
        xs.append(estimator.get_state_estimate())
        if step["done"]:
            break

    x = np.asarray(xs, dtype=float)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, series=x)
    print(out)


if __name__ == "__main__":
    main()
