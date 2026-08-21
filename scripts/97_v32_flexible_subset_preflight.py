#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.power_projector import PowerConstraintsV2, PowerProjector  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.warmup_state import SensorRuntime  # noqa: E402


def feasible_masks(sensors: list, *, budget: float, startup_budget: float) -> np.ndarray:
    constraints = PowerConstraintsV2(
        max_active=None,
        per_step_budget=float(budget),
        startup_peak_budget=float(startup_budget),
        required_sensor_ids=(),
        coverage_groups=(),
    )
    projector = PowerProjector(sensors, constraints)
    runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for value in range(1 << len(sensors)):
        desired = np.asarray([(value >> idx) & 1 for idx in range(len(sensors))], dtype=bool)
        result = projector.project_mask(desired, runtimes)
        key = tuple(int(item) for item in result.selected_mask)
        unique[key] = result.selected_mask.astype(bool)
    return np.asarray(list(unique.values()), dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit power-only subset geometry for the flexible PD-PPO scene.")
    parser.add_argument(
        "--sensor-cfg",
        default="configs/sensors/windblown_sensors_flexible_subset_v1.yaml",
    )
    parser.add_argument("--budget", type=float, default=1.35)
    parser.add_argument("--startup-peak-budget", type=float, default=1.65)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sensors = load_sensor_specs(ROOT / args.sensor_cfg)
    masks = feasible_masks(
        sensors,
        budget=float(args.budget),
        startup_budget=float(args.startup_peak_budget),
    )
    costs = np.asarray([float(spec.power_cost) for spec in sensors], dtype=float)
    peaks = np.asarray([float(max(spec.power_cost, spec.startup_peak_power)) for spec in sensors], dtype=float)
    cardinalities = np.sum(masks, axis=1).astype(int)
    full_open_cost = float(np.sum(costs))
    full_open_peak = float(np.sum(peaks))
    payload = {
        "sensor_cfg": str(args.sensor_cfg),
        "sensor_ids": [str(spec.sensor_id) for spec in sensors],
        "effective_costs": costs.tolist(),
        "startup_costs": peaks.tolist(),
        "budget": float(args.budget),
        "startup_peak_budget": float(args.startup_peak_budget),
        "candidate_mask_count": int(len(masks)),
        "cardinality_counts": {
            str(key): int(value) for key, value in sorted(Counter(cardinalities.tolist()).items())
        },
        "channel_candidate_counts": {
            str(spec.sensor_id): int(np.sum(masks[:, idx])) for idx, spec in enumerate(sensors)
        },
        "all_single_channels_feasible": bool(all(np.any(np.all(masks == np.eye(len(sensors), dtype=bool)[idx], axis=1)) for idx in range(len(sensors)))),
        "empty_mask_feasible": bool(np.any(np.all(~masks, axis=1))),
        "full_open_cost": full_open_cost,
        "full_open_peak": full_open_peak,
        "full_open_feasible": bool(
            full_open_cost <= float(args.budget) + 1e-12
            and full_open_peak <= float(args.startup_peak_budget) + 1e-12
        ),
        "masks": [
            {
                "sensor_ids": [str(sensors[idx].sensor_id) for idx in np.flatnonzero(mask)],
                "cardinality": int(np.sum(mask)),
                "steady_cost": float(np.dot(mask.astype(float), costs)),
                "cold_start_cost": float(np.dot(mask.astype(float), peaks)),
            }
            for mask in masks
        ],
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
