#!/usr/bin/env python3
"""Audit V607 dynamic-resource rollouts without counting the logger backbone.

This is a result-audit utility, not a training or policy-selection script.
It recomputes dynamic optional-channel cost from the frozen resource trace and
separates the mandatory CR1000Xe logger from selectable sensor behaviour.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


OPTIONAL_COST_COLUMNS = {
    "met_station_core": "resource_effective_power_met_station_core",
    "radiometer_basic": "resource_effective_power_radiometer_basic",
    "surface_temp_ir": "resource_effective_power_surface_temp_ir",
    "laser_disdrometer": "resource_effective_power_laser_disdrometer",
    "fc4_flux": "resource_effective_power_fc4_flux",
}
MANDATORY_ID = "cr1000xe_backbone"


def _duty_counts(selected: np.ndarray, optional_indices: list[int]) -> tuple[int, int, int]:
    duty = selected[:, optional_indices].mean(axis=0)
    return (
        int(np.sum(duty >= 0.95)),
        int(np.sum(duty <= 0.05)),
        int(np.sum((duty > 0.05) & (duty < 0.95))),
    )


def audit_one(seed: int, seed_dir: Path, trace_path: Path, budget: float) -> list[dict[str, object]]:
    trace = pd.read_csv(trace_path).set_index("time_idx")
    rows: list[dict[str, object]] = []
    for rollout_path in sorted(seed_dir.glob("rollout_*.npz")):
        z = np.load(rollout_path, allow_pickle=True)
        sensor_ids = [str(x) for x in z["sensor_ids"]]
        selected = np.asarray(z["selected_masks"], dtype=int)
        step_indices = np.asarray(z["step_indices"], dtype=int)
        selected_trace = trace.loc[step_indices]
        optional_indices = [i for i, sid in enumerate(sensor_ids) if sid in OPTIONAL_COST_COLUMNS]
        mandatory_indices = [i for i, sid in enumerate(sensor_ids) if sid == MANDATORY_ID]
        optional_ids = [sensor_ids[i] for i in optional_indices]

        dynamic_cost = np.zeros(len(selected), dtype=float)
        for i in optional_indices:
            dynamic_cost += selected[:, i] * selected_trace[OPTIONAL_COST_COLUMNS[sensor_ids[i]]].to_numpy()
        feasible_counts = np.zeros(len(selected), dtype=int)
        for j, (_, trace_row) in enumerate(selected_trace.iterrows()):
            costs = np.asarray([trace_row[OPTIONAL_COST_COLUMNS[sid]] for sid in optional_ids], dtype=float)
            feasible_counts[j] = sum(float(np.dot(mask, costs)) <= budget + 1e-9 for mask in itertools.product((0, 1), repeat=len(costs)))
        on, off, mid = _duty_counts(selected, optional_indices)
        rows.append(
            {
                "seed": seed,
                "policy": str(z["policy"][0]) if "policy" in z else rollout_path.stem.removeprefix("rollout_"),
                "steps": int(len(selected)),
                "optional_sensor_count": len(optional_indices),
                "mandatory_sensor_ids": ";".join(sensor_ids[i] for i in mandatory_indices),
                "mandatory_always_on_count": int(sum(np.all(selected[:, i] == 1) for i in mandatory_indices)),
                "optional_always_on_count": on,
                "optional_always_off_count": off,
                "optional_mid_duty_count": mid,
                "optional_selected_count_mean": float(selected[:, optional_indices].sum(axis=1).mean()),
                "optional_unique_mask_count": int(np.unique(selected[:, optional_indices], axis=0).shape[0]),
                "dynamic_cost_mean_w": float(dynamic_cost.mean()),
                "dynamic_cost_max_w": float(dynamic_cost.max()),
                "dynamic_cost_violations": int(np.sum(dynamic_cost > budget + 1e-9)),
                "feasible_mask_count_min": int(feasible_counts.min()),
                "feasible_mask_count_median": float(np.median(feasible_counts)),
                "feasible_mask_count_max": int(feasible_counts.max()),
                "selected_masks_feasible": bool(np.all(dynamic_cost <= budget + 1e-9)),
                "warmup_abort_count": int(np.asarray(z["warmup_abort_count"]).reshape(-1)[0]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=2.15)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for seed_dir in sorted(args.report.glob("seed*")):
        digits = "".join(ch for ch in seed_dir.name if ch.isdigit())
        if not digits:
            continue
        seed = int(digits[-4:])
        trace = args.trace_dir / f"resource_trace_seed{seed}.csv"
        if trace.exists():
            records.extend(audit_one(seed, seed_dir, trace, args.budget))
    result = pd.DataFrame(records).sort_values(["seed", "policy"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    md = args.output.with_suffix(".md")
    lines = [
        "# V607 dynamic-resource constraint audit",
        "",
        "This audit recomputes optional-channel effective cost from the frozen resource trace. The mandatory CR1000Xe logger is reported separately and is excluded from optional duty counts.",
        "",
        f"Budget: `{args.budget:.2f} W` optional effective-resource budget.",
        "",
        result.to_markdown(index=False),
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
