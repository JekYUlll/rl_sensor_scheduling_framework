#!/usr/bin/env python3
"""Truth-only occupancy and feasible-frontier screen for a resource trace."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2.sensor_spec import load_sensor_specs  # noqa: E402


CHANNELS = ("met_station_core", "radiometer_basic", "surface_temp_ir", "laser_disdrometer", "fc4_flux")
BACKBONE = 0.4104
FINAL_STARTS = (82600, 83000, 84800, 85500)
FINAL_STEPS = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sensor-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=2.50)
    args = parser.parse_args()
    trace = pd.read_csv(args.trace)
    specs = {spec.sensor_id: spec for spec in load_sensor_specs(str(args.sensor_config))}
    startup = {name: float(specs[name].startup_peak_power) for name in CHANNELS}
    final_idx = np.concatenate([np.arange(start, start + FINAL_STEPS) for start in FINAL_STARTS])
    frame = trace.iloc[final_idx].reset_index(drop=True)
    rows = []
    for state, group in frame.groupby([
        "resource_heater_on_met_station_core",
        "resource_heater_on_laser_disdrometer",
    ], sort=True):
        counts = []
        for mask in itertools.product((0, 1), repeat=len(CHANNELS)):
            selected = [name for name, bit in zip(CHANNELS, mask) if bit]
            steady = BACKBONE + sum(float(group[f"resource_effective_power_{name}"].iloc[0]) for name in selected)
            peak = BACKBONE + sum(startup[name] for name in selected)
            if steady <= args.budget + 1e-12 and peak <= args.budget + 1e-12:
                counts.append("".join(str(bit) for bit in mask))
        rows.append({
            "heater_state": "".join(str(int(x)) for x in state),
            "rows": int(len(group)),
            "fraction": float(len(group) / len(frame)),
            "feasible_mask_count": len(counts),
            "feasible_masks": counts,
        })
    result = {
        "trace": str(args.trace),
        "budget": float(args.budget),
        "final_rows": int(len(frame)),
        "heater_states": rows,
        "all_four_states_supported": len(rows) == 4,
        "frontier_changes": len({tuple(row["feasible_masks"]) for row in rows}) > 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "seed": args.trace.stem.replace("resource_seed", ""),
        "all_four_states_supported": result["all_four_states_supported"],
        "frontier_changes": result["frontier_changes"],
        "feasible_counts": {row["heater_state"]: row["feasible_mask_count"] for row in rows},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
