"""Audit the documented physical resource envelope without running a scene.

This is a structural gate for the flexible-subset redesign. It enumerates all
subsets of the five documented selectable instruments, always including the
CR1000Xe logger backbone, and compares their documented loads with the design
envelope. It deliberately does not invent a renewable time series or a heater
duty cycle.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from v2.entity_resource_envelope import (
    BATTERY_CAPACITY_WH,
    CONTROLLER_CURRENT_A,
    CONTROLLER_POWER_W,
    CONTROLLER_VOLTAGE_V,
    FIXED_AUXILIARY_POWER_W,
    PV_RATED_W,
    WIND_RATED_W,
    CHANNEL_DEVICE,
    rows,
)


def write_outputs(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = rows()
    with (output_dir / "resource_envelope_subsets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

    all_channels = next(row for row in data if row["selected_count"] == len(CHANNEL_DEVICE))
    summary = {
        "mapping_scope": "five selectable measurement channels plus mandatory CR1000Xe backbone",
        "selectable_channel_count": len(CHANNEL_DEVICE),
        "enumerated_subset_count": len(data),
        "battery_capacity_wh": BATTERY_CAPACITY_WH,
        "controller_voltage_v": CONTROLLER_VOLTAGE_V,
        "controller_current_a": CONTROLLER_CURRENT_A,
        "controller_power_w": CONTROLLER_POWER_W,
        "pv_rated_w": PV_RATED_W,
        "wind_rated_w": WIND_RATED_W,
        "fixed_auxiliary_power_w": FIXED_AUXILIARY_POWER_W,
        "all_channels": all_channels,
        "all_channels_steady_feasible": bool(all_channels["controller_steady_feasible"]),
        "all_channels_heater_peak_feasible": bool(all_channels["controller_heater_peak_feasible"]),
        "all_channels_steady_battery_hours": all_channels["battery_hours_at_steady_load"],
        "caveat": "Design-envelope calculation; includes the three listed fixed auxiliary loads but excludes conversion losses, other fixed loads, and unverified heater duty or renewable traces.",
    }
    (output_dir / "resource_envelope_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_outputs(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
