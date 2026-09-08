"""Append a predeclared fixed heater load to a truth CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2.entity_external_load import build_hysteretic_heater_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-truth", type=Path, required=True)
    parser.add_argument("--output-truth", type=Path, required=True)
    parser.add_argument("--surface-temperature-column", default="snow_surface_temperature_c")
    parser.add_argument("--output-column", default="fixed_external_power_w")
    parser.add_argument("--on-threshold-c", type=float, default=-25.0)
    parser.add_argument("--off-threshold-c", type=float, default=-23.0)
    parser.add_argument("--heater-power-w", type=float, default=600.0)
    args = parser.parse_args()

    truth = pd.read_csv(args.input_truth)
    if args.surface_temperature_column not in truth.columns:
        raise ValueError(f"missing surface temperature column: {args.surface_temperature_column}")
    profile = build_hysteretic_heater_profile(
        truth[args.surface_temperature_column].to_numpy(dtype=float),
        on_threshold_c=float(args.on_threshold_c),
        off_threshold_c=float(args.off_threshold_c),
        heater_power_w=float(args.heater_power_w),
    )
    truth[args.output_column] = profile
    args.output_truth.parent.mkdir(parents=True, exist_ok=True)
    truth.to_csv(args.output_truth, index=False)
    summary = {
        "input_truth": str(args.input_truth),
        "output_truth": str(args.output_truth),
        "surface_temperature_column": args.surface_temperature_column,
        "output_column": args.output_column,
        "on_threshold_c": float(args.on_threshold_c),
        "off_threshold_c": float(args.off_threshold_c),
        "heater_power_w": float(args.heater_power_w),
        "rows": int(len(truth)),
        "heater_on_fraction": float(np.mean(profile > 0.0)),
        "heater_power_mean_w": float(profile.mean()),
        "heater_power_max_w": float(profile.max()),
        "policy_controllable": False,
    }
    summary_path = args.output_truth.with_suffix(".load_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
