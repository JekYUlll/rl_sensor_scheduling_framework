#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_lengths(flags: np.ndarray) -> list[tuple[bool, int]]:
    values = np.asarray(flags, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    runs: list[tuple[bool, int]] = []
    start = 0
    current = bool(values[0])
    for idx in range(1, values.size):
        value = bool(values[idx])
        if value != current:
            runs.append((current, int(idx - start)))
            start = idx
            current = value
    runs.append((current, int(values.size - start)))
    return runs


def describe_lengths(lengths: list[int]) -> dict[str, float | int]:
    arr = np.asarray(lengths, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def lead_trigger(flags: np.ndarray, lead_steps: int) -> np.ndarray:
    values = np.asarray(flags, dtype=bool).reshape(-1)
    out = np.zeros(values.size, dtype=bool)
    lead = max(0, int(lead_steps))
    for idx in range(values.size):
        out[idx] = bool(np.any(values[idx : min(values.size, idx + lead + 1)]))
    return out


def max_drawdown(cost: np.ndarray, harvest: float) -> tuple[float, float]:
    net = np.asarray(cost, dtype=float).reshape(-1) - float(harvest)
    cumulative = np.cumsum(net)
    prior_min = np.minimum.accumulate(np.r_[0.0, cumulative[:-1]])
    return float(np.max(cumulative - prior_min)), float(cumulative[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate normalized energy-account parameters from V3.1 event clusters.")
    parser.add_argument("--truth-csv", required=True)
    parser.add_argument("--event-column", default="event_flag")
    parser.add_argument("--eval-start-indices", nargs="*", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--lead-steps", type=int, default=4)
    parser.add_argument("--calm-cost", type=float, default=0.32)
    parser.add_argument("--event-laser-cost", type=float, default=1.16)
    parser.add_argument("--static-snow-cost", type=float, default=0.82)
    parser.add_argument("--static-laser-cost", type=float, default=1.16)
    parser.add_argument("--harvest-grid", nargs="*", type=float, default=[0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    truth = pd.read_csv(args.truth_csv)
    flags = truth[str(args.event_column)].astype(bool).to_numpy()
    event_runs = [length for value, length in run_lengths(flags) if value]
    calm_runs = [length for value, length in run_lengths(flags) if not value]
    trigger = lead_trigger(flags, int(args.lead_steps))
    trigger_runs = [length for value, length in run_lengths(trigger) if value]
    nontrigger_runs = [length for value, length in run_lengths(trigger) if not value]

    dynamic_cost = np.where(trigger, float(args.event_laser_cost), float(args.calm_cost))
    static_snow_cost = np.full(flags.size, float(args.static_snow_cost), dtype=float)
    static_laser_cost = np.full(flags.size, float(args.static_laser_cost), dtype=float)

    rows = []
    for harvest in [float(x) for x in args.harvest_grid]:
        row = {
            "harvest": float(harvest),
            "dynamic_full_drawdown": max_drawdown(dynamic_cost, harvest)[0],
            "dynamic_full_end_net": max_drawdown(dynamic_cost, harvest)[1],
            "static_snow_full_drawdown": max_drawdown(static_snow_cost, harvest)[0],
            "static_laser_full_drawdown": max_drawdown(static_laser_cost, harvest)[0],
        }
        starts = args.eval_start_indices or []
        for start in starts:
            end = min(flags.size, int(start) + int(args.eval_steps))
            prefix = f"eval_{int(start)}"
            row[f"{prefix}_trigger_fraction"] = float(np.mean(trigger[int(start) : end]))
            row[f"{prefix}_dynamic_drawdown"] = max_drawdown(dynamic_cost[int(start) : end], harvest)[0]
            row[f"{prefix}_static_snow_drawdown"] = max_drawdown(static_snow_cost[int(start) : end], harvest)[0]
            row[f"{prefix}_static_laser_drawdown"] = max_drawdown(static_laser_cost[int(start) : end], harvest)[0]
        rows.append(row)

    trigger_fraction = float(np.mean(trigger))
    event_fraction = float(np.mean(flags))
    summary = {
        "truth_csv": str(args.truth_csv),
        "event_fraction": event_fraction,
        "event_runs": describe_lengths(event_runs),
        "calm_runs": describe_lengths(calm_runs),
        "lead_steps": int(args.lead_steps),
        "trigger_fraction": trigger_fraction,
        "trigger_runs": describe_lengths(trigger_runs),
        "nontrigger_runs": describe_lengths(nontrigger_runs),
        "costs": {
            "calm_cost": float(args.calm_cost),
            "event_laser_cost": float(args.event_laser_cost),
            "static_snow_cost": float(args.static_snow_cost),
            "static_laser_cost": float(args.static_laser_cost),
        },
        "average_costs": {
            "dynamic_lead_laser": float(np.mean(dynamic_cost)),
            "static_snow": float(args.static_snow_cost),
            "static_laser": float(args.static_laser_cost),
        },
        "harvest_rows": rows,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
