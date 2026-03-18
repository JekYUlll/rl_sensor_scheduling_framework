#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_cases.windblown_case.generator_adapter import generate_windblown_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", default="configs/env/windblown_case.yaml")
    parser.add_argument("--sensor_cfg", default="configs/sensors/windblown_sensors.yaml")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--out", default="data/generated/windblown_truth.csv")
    args = parser.parse_args()

    out = generate_windblown_csv(args.env_cfg, args.sensor_cfg, args.steps, args.out)
    print(out)


if __name__ == "__main__":
    main()
