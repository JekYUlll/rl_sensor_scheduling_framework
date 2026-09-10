#!/usr/bin/env python3
"""Build split specialist affinities for transport, particles, thermal and radiation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEAD_STEPS = 6


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def build(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = frame.copy()
    drivers = {}
    for name in ("transport", "particle", "thermal"):
        col = f"agent_context_shared_{name}_driver"
        if col not in out:
            raise ValueError(f"missing driver column: {col}")
        values = out[col].to_numpy(float)
        lagged = np.zeros_like(values)
        lagged[LEAD_STEPS:] = values[:-LEAD_STEPS]
        drivers[name] = np.clip(lagged, 0.0, 1.0)
    solar = out["agent_context_nowcast_solar_radiation_wm2"].to_numpy(float)
    solar_lagged = np.zeros_like(solar)
    solar_lagged[LEAD_STEPS:] = solar[:-LEAD_STEPS]
    drivers["radiation"] = sigmoid((solar_lagged - 20.0) / 40.0)
    relations = {
        "fc4_flux": ("transport", 0.10),
        "laser_disdrometer": ("particle", 0.10),
        "surface_temp_ir": ("thermal", 0.15),
        "radiometer_basic": ("radiation", 0.15),
    }
    for channel, (driver_name, floor) in relations.items():
        out[f"agent_context_quality_{channel}"] = floor + (1.0 - floor) * drivers[driver_name]
        out[f"generator_affinity_{channel}"] = drivers[driver_name]
    out["agent_context_quality_cr1000xe_backbone"] = 1.0
    return out, {
        "generator": Path(__file__).name,
        "lead_steps": LEAD_STEPS,
        "quality_convention": "higher score means lower observation noise",
        "specialist_relation": {channel: driver for channel, (driver, _) in relations.items()},
        "radiation_driver": "sigmoid((lagged_solar_radiation_wm2 - 20) / 40)",
        "future_targets_used": False,
        "event_labels_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out, metadata = build(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
