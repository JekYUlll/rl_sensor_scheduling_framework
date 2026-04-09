#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _load_targets(env_cfg: Path, target_set: str) -> list[str]:
    with env_cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if target_set == "primary":
        targets = cfg.get("reward_target_columns", [])
    elif target_set == "forecast":
        targets = cfg.get("forecast_target_columns", [])
    else:
        raise ValueError(f"unsupported target_set={target_set}")
    return [str(t) for t in targets]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all posthoc prediction and activation plots for one run tag")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--target", default="snow_mass_flux_kg_m2_s")
    parser.add_argument("--target-set", choices=["single", "primary", "forecast"], default="single")
    parser.add_argument("--env-cfg", default="configs/env/windblown_case.yaml")
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors.yaml")
    parser.add_argument("--max-points", type=int, default=300)
    parser.add_argument("--timeline-start", type=int, default=0)
    parser.add_argument("--timeline-end", type=int, default=300)
    args = parser.parse_args()

    targets = [args.target]
    if args.target_set != "single":
        targets = _load_targets(ROOT / args.env_cfg, args.target_set)

    for target in targets:
        for horizon in (1, 2, 3):
            _run(
                [
                    sys.executable,
                    "scripts/07_plot_scheduler_prediction_curves.py",
                    "--run-tag",
                    args.run_tag,
                    "--model",
                    "all",
                    "--target",
                    target,
                    "--horizon",
                    str(horizon),
                    "--max-points",
                    str(args.max_points),
                ]
            )

        _run(
            [
                sys.executable,
                "scripts/08_plot_sensor_activation_timelines.py",
                "--run-tag",
                args.run_tag,
                "--target",
                target,
                "--sensor-cfg",
                args.sensor_cfg,
                "--start",
                str(args.timeline_start),
                "--end",
                str(args.timeline_end),
            ]
        )


if __name__ == "__main__":
    main()
