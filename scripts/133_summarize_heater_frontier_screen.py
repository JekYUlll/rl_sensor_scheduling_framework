"""Summarize V600-V602 occupancy and frontier persistence across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def runs(values: pd.Series) -> list[int]:
    lengths: list[int] = []
    start = 0
    array = values.astype(bool).to_numpy()
    for idx in range(1, len(array) + 1):
        if idx == len(array) or array[idx] != array[start]:
            lengths.append(idx - start)
            start = idx
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        trace = pd.read_csv(args.root / f"resource_trace_seed{seed}.csv")
        frontier = pd.read_csv(args.root / f"seed{seed}" / f"frontier_seed{seed}.csv")
        gmx_runs = runs(trace["resource_heater_on_met_station_core"])
        parsivel_runs = runs(trace["resource_heater_on_laser_disdrometer"])
        effective_sets = [set(str(value).split(";") if value else []) for value in frontier["effective_feasible_bitsets"]]
        common = set.intersection(*effective_sets) if effective_sets else set()
        frontier_values = frontier["effective_feasible_bitsets"].to_numpy()
        frontier_changes = int((frontier_values[1:] != frontier_values[:-1]).sum())
        rows.append({
            "seed": seed,
            "rows": len(trace),
            "gmx_heater_fraction": float(trace["resource_heater_on_met_station_core"].mean()),
            "parsivel_heater_fraction": float(trace["resource_heater_on_laser_disdrometer"].mean()),
            "gmx_transition_rate": float(trace["resource_heater_on_met_station_core"].astype(int).diff().fillna(0).ne(0).mean()),
            "parsivel_transition_rate": float(trace["resource_heater_on_laser_disdrometer"].astype(int).diff().fillna(0).ne(0).mean()),
            "gmx_median_run_length": float(np.median(gmx_runs)),
            "parsivel_median_run_length": float(np.median(parsivel_runs)),
            "gmx_p90_run_length": float(np.percentile(gmx_runs, 90)),
            "parsivel_p90_run_length": float(np.percentile(parsivel_runs, 90)),
            "effective_frontier_change_rate": float(frontier_changes / max(1, len(frontier) - 1)),
            "effective_common_feasible_mask_count": len(common),
            "effective_feasible_count_min": int(frontier["effective_feasible_count"].min()),
            "effective_feasible_count_max": int(frontier["effective_feasible_count"].max()),
            "absolute_frontier_unique": int(frontier["absolute_feasible_bitsets"].nunique()),
            "effective_frontier_unique": int(frontier["effective_feasible_bitsets"].nunique()),
        })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = {
        "seeds": args.seeds,
        "seed_count": len(rows),
        "all_absolute_frontiers_constant": bool((result["absolute_frontier_unique"] == 1).all()),
        "all_effective_frontiers_change": bool((result["effective_frontier_unique"] > 1).all()),
        "mean_gmx_heater_fraction": float(result["gmx_heater_fraction"].mean()),
        "mean_parsivel_heater_fraction": float(result["parsivel_heater_fraction"].mean()),
        "mean_effective_frontier_change_rate": float(result["effective_frontier_change_rate"].mean()),
        "mean_effective_feasible_count_min": float(result["effective_feasible_count_min"].mean()),
        "mean_effective_feasible_count_max": float(result["effective_feasible_count_max"].mean()),
        "status": "physics_screen_only; no forecast geometry or PPO evidence",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
