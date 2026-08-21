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


def _load_env_cfg(env_cfg: Path) -> dict:
    with env_cfg.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_targets(env_cfg: Path, target_set: str) -> list[str]:
    cfg = _load_env_cfg(env_cfg)
    if target_set == "primary":
        targets = cfg.get("reward_target_columns", [])
    elif target_set == "forecast":
        targets = cfg.get("forecast_target_columns", [])
    else:
        raise ValueError(f"unsupported target_set={target_set}")
    return [str(t) for t in targets]


def _prediction_targets_for(target: str, forecast_targets: list[str]) -> list[str]:
    if target in forecast_targets:
        return [target]
    if target == "wind_direction_deg":
        return [name for name in ("wind_dir_sin", "wind_dir_cos") if name in forecast_targets]
    return []


def _timeline_target_for(target: str) -> str:
    if target in {"wind_dir_sin", "wind_dir_cos"}:
        return "wind_direction_deg"
    return target


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
    env_cfg_path = ROOT / args.env_cfg
    env_cfg = _load_env_cfg(env_cfg_path)
    forecast_targets = [str(t) for t in env_cfg.get("forecast_target_columns", [])]
    if args.target_set != "single":
        targets = _load_targets(env_cfg_path, args.target_set)

    for target in targets:
        prediction_targets = _prediction_targets_for(str(target), forecast_targets)
        if not prediction_targets:
            print(f"[skip] no forecast prediction target for {target!r}")
        for prediction_target in prediction_targets:
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
                        prediction_target,
                        "--horizon",
                        str(horizon),
                        "--max-points",
                        str(args.max_points),
                    ]
                )

        timeline_target = _timeline_target_for(str(target))
        if timeline_target in {_timeline_target_for(str(t)) for t in targets[: targets.index(target)]}:
            continue
        try:
            _run(
                [
                    sys.executable,
                    "scripts/08_plot_sensor_activation_timelines.py",
                    "--run-tag",
                    args.run_tag,
                    "--target",
                    timeline_target,
                    "--sensor-cfg",
                    args.sensor_cfg,
                    "--start",
                    str(args.timeline_start),
                    "--end",
                    str(args.timeline_end),
                ]
            )
        except SystemExit as exc:
            if int(exc.code) != 0:
                print(f"[skip] activation timeline target {timeline_target!r} is unavailable")


if __name__ == "__main__":
    main()
