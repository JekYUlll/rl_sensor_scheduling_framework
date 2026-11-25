"""Validate and materialize the V600 entity heater/resource manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {
    "met_station_core",
    "laser_disdrometer",
    "radiometer_basic",
    "surface_temp_ir",
    "fc4_flux",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    channels = {row["channel_id"]: row for row in payload["channels"]}
    if set(channels) != REQUIRED:
        raise ValueError(f"manifest channel mismatch: {sorted(channels)}")
    if payload["development_effective_budget"] <= 0:
        raise ValueError("development effective budget must be positive")
    for row in channels.values():
        for key in ("base_power_w", "effective_base", "heater_increment_w", "heater_increment_effective"):
            if float(row[key]) < 0:
                raise ValueError(f"negative resource value: {row['channel_id']} {key}")
    output = dict(payload)
    output["validation"] = {
        "required_channels_present": True,
        "channel_count": len(channels),
        "cnf4_mapped_to_lps10": False,
        "truth_targets_changed": False,
        "event_labels_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["validation"], sort_keys=True))


if __name__ == "__main__":
    main()
