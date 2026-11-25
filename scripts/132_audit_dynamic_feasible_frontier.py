"""Enumerate all 32 optional subsets under V602 resource envelopes."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    truth = pd.read_csv(args.input)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    channels = [row["channel_id"] for row in manifest["channels"]]
    records = []
    for bits in itertools.product((0, 1), repeat=len(channels)):
        selected = [c for c, bit in zip(channels, bits) if bit]
        bitset = "".join(map(str, bits))
        physical = sum(float(truth[f"resource_power_w_{c}"].iloc[0]) * bit for c, bit in zip(channels, bits))
        effective = sum(float(truth[f"resource_effective_power_{c}"].iloc[0]) * bit for c, bit in zip(channels, bits))
        # Per-row values are added below; this first record is only a template.
        records.append((bitset, selected))
    frontier_rows = []
    for idx, row in truth.iterrows():
        absolute_feasible = []
        effective_feasible = []
        for bitset, selected in records:
            physical = sum(float(row[f"resource_power_w_{c}"]) for c in selected) + float(manifest["mandatory_backbone"]["steady_power_w"])
            effective = sum(float(row[f"resource_effective_power_{c}"]) for c in selected)
            if physical <= float(manifest["absolute_controller_budget_w"]):
                absolute_feasible.append(bitset)
            if effective <= float(manifest["development_effective_budget"]):
                effective_feasible.append(bitset)
        frontier_rows.append({
            "time_idx": int(row["time_idx"]) if "time_idx" in row else int(idx),
            "absolute_feasible_count": len(absolute_feasible),
            "effective_feasible_count": len(effective_feasible),
            "absolute_feasible_bitsets": ";".join(absolute_feasible),
            "effective_feasible_bitsets": ";".join(effective_feasible),
        })
    frontier = pd.DataFrame(frontier_rows)
    summary = {
        "seed": args.seed, "rows": len(frontier), "optional_channel_count": len(channels),
        "mask_count": 2 ** len(channels),
        "absolute_feasible_count_min": int(frontier.absolute_feasible_count.min()),
        "absolute_feasible_count_max": int(frontier.absolute_feasible_count.max()),
        "effective_feasible_count_min": int(frontier.effective_feasible_count.min()),
        "effective_feasible_count_max": int(frontier.effective_feasible_count.max()),
        "absolute_frontier_unique": int(frontier.absolute_feasible_bitsets.nunique()),
        "effective_frontier_unique": int(frontier.effective_feasible_bitsets.nunique()),
        "gmx_heater_fraction": float(truth["resource_heater_on_met_station_core"].mean()),
        "parsivel_heater_fraction": float(truth["resource_heater_on_laser_disdrometer"].mean()),
        "absolute_all_optional_feasible_fraction": float(truth["resource_absolute_feasible_all_optional"].mean()),
        "effective_all_optional_feasible_fraction": float(truth["resource_effective_feasible_all_optional"].mean()),
        "absolute_budget_w": float(manifest["absolute_controller_budget_w"]),
        "development_effective_budget": float(manifest["development_effective_budget"]),
        "screen_status": "diagnostic_only_until_installed_telemetry",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(args.output_dir / f"frontier_seed{args.seed}.csv", index=False)
    (args.output_dir / f"frontier_seed{args.seed}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
