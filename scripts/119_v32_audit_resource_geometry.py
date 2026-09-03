#!/usr/bin/env python3
"""Enumerate the exact steady/startup resource-geometry breakpoints.

This audit is independent of policy performance. It reads the physical-channel
configuration, enumerates all subsets, and reports how the feasible family
changes when either power limit crosses an actual subset cost.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v2.sensor_spec import load_sensor_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-cfg", type=Path, required=True)
    parser.add_argument("--steady-budget", type=float, required=True)
    parser.add_argument("--startup-budget", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def enumerate_subsets(sensor_cfg: Path) -> pd.DataFrame:
    sensors = load_sensor_specs(sensor_cfg)
    rows: list[dict[str, object]] = []
    for bits in itertools.product((False, True), repeat=len(sensors)):
        selected = [spec for spec, active in zip(sensors, bits, strict=True) if active]
        rows.append(
            {
                "mask_bits": "".join("1" if active else "0" for active in bits),
                "sensor_ids": "+".join(spec.sensor_id for spec in selected),
                "cardinality": len(selected),
                "steady_cost": sum(float(spec.power_cost) for spec in selected),
                "cold_start_cost": sum(float(spec.startup_peak_power) for spec in selected),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["steady_cost", "cold_start_cost", "cardinality", "mask_bits"],
        kind="stable",
    ).reset_index(drop=True)


def breakpoint_scan(
    subsets: pd.DataFrame,
    *,
    variable: str,
    fixed_limit_column: str,
    fixed_limit: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = subsets[subsets[fixed_limit_column] <= fixed_limit + 1.0e-12]
    previous: set[str] = set()
    for breakpoint in sorted(float(value) for value in eligible[variable].unique()):
        current_rows = eligible[eligible[variable] <= breakpoint + 1.0e-12]
        current = set(str(value) for value in current_rows.mask_bits)
        added = sorted(current - previous)
        cardinalities = current_rows.cardinality.value_counts().sort_index()
        rows.append(
            {
                "scanned_limit": variable,
                "breakpoint": breakpoint,
                "fixed_limit": fixed_limit_column,
                "fixed_limit_value": fixed_limit,
                "feasible_mask_count": len(current),
                "max_cardinality": int(current_rows.cardinality.max()),
                "cardinality_counts": json.dumps(
                    {str(int(key)): int(value) for key, value in cardinalities.items()},
                    sort_keys=True,
                ),
                "new_mask_bits": ";".join(added),
            }
        )
        previous = current
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    subsets = enumerate_subsets(args.sensor_cfg)
    subsets["feasible_at_current_limits"] = (
        (subsets.steady_cost <= args.steady_budget + 1.0e-12)
        & (subsets.cold_start_cost <= args.startup_budget + 1.0e-12)
    )
    subsets.to_csv(args.out_dir / "all_subset_costs.csv", index=False)

    steady_scan = breakpoint_scan(
        subsets,
        variable="steady_cost",
        fixed_limit_column="cold_start_cost",
        fixed_limit=args.startup_budget,
    )
    startup_scan = breakpoint_scan(
        subsets,
        variable="cold_start_cost",
        fixed_limit_column="steady_cost",
        fixed_limit=args.steady_budget,
    )
    phase = pd.concat([steady_scan, startup_scan], ignore_index=True)
    phase.to_csv(args.out_dir / "resource_breakpoints.csv", index=False)

    feasible = subsets[subsets.feasible_at_current_limits]
    summary = {
        "sensor_cfg": str(args.sensor_cfg),
        "sensor_ids": [spec.sensor_id for spec in load_sensor_specs(args.sensor_cfg)],
        "steady_budget": float(args.steady_budget),
        "startup_budget": float(args.startup_budget),
        "all_subset_count": int(len(subsets)),
        "current_feasible_mask_count": int(len(feasible)),
        "current_cardinality_counts": {
            str(int(key)): int(value)
            for key, value in feasible.cardinality.value_counts().sort_index().items()
        },
        "current_max_cardinality": int(feasible.cardinality.max()),
        "steady_breakpoints_with_fixed_startup": [float(value) for value in steady_scan.breakpoint],
        "startup_breakpoints_with_fixed_steady": [float(value) for value in startup_scan.breakpoint],
    }
    (args.out_dir / "resource_geometry_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
