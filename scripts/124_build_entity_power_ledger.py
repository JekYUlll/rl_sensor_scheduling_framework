#!/usr/bin/env python3
"""Emit the documented entity-level power ledger and a smoke calculation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.v2.entity_energy import ledger_rows, system_power_w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = ledger_rows()
    with (args.output_dir / "entity_power_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    smoke = {
        "all_documented_channels": system_power_w(list({r["channel_id"] for r in rows[:-1]})),
        "parsivel_heater_active": system_power_w(["met_station_core", "laser_disdrometer"], heater_on=True),
        "mapping_scope": "five selectable measurement channels plus mandatory CR1000Xe backbone",
    }
    (args.output_dir / "entity_power_ledger.json").write_text(
        json.dumps(smoke, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(smoke, indent=2))


if __name__ == "__main__":
    main()
