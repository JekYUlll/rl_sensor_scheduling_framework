#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.power_projector import PowerConstraintsV2, PowerProjector  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.warmup_state import SensorRuntime  # noqa: E402


def resolve_sensor_cfg(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def enumerate_feasible_subsets(
    *,
    sensor_cfg: Path,
    budgets: list[float],
    startup_peak_budget: float | None,
    max_active: int | None,
    required_sensors: tuple[str, ...],
    out_dir: Path,
) -> pd.DataFrame:
    sensors = load_sensor_specs(sensor_cfg)
    rows: list[dict[str, object]] = []
    for budget in budgets:
        constraints = PowerConstraintsV2(
            max_active=max_active,
            per_step_budget=float(budget),
            startup_peak_budget=startup_peak_budget,
            required_sensor_ids=required_sensors,
            coverage_groups=(),
        )
        projector = PowerProjector(sensors, constraints)
        runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
        for size in range(0, len(sensors) + 1):
            if max_active is not None and size > int(max_active):
                continue
            for combo in combinations(range(len(sensors)), size):
                desired = [False] * len(sensors)
                for idx in combo:
                    desired[idx] = True
                try:
                    result = projector.project_mask(desired, runtimes)
                except ValueError:
                    continue
                selected = tuple(int(i) for i, flag in enumerate(result.selected_mask) if bool(flag))
                if len(selected) != len(set(selected)):
                    continue
                selected_ids = tuple(sensors[idx].sensor_id for idx in selected)
                rows.append(
                    {
                        "budget": float(budget),
                        "sensor_count": int(len(selected_ids)),
                        "steady_power": float(result.steady_power),
                        "cold_start_peak_power": float(result.peak_power),
                        "sensor_ids": "|".join(selected_ids),
                        "has_laser": "laser_disdrometer" in selected_ids,
                        "has_fc4": "fc4_flux" in selected_ids,
                        "has_snow_particle_counter": "snow_particle_counter" in selected_ids,
                        "has_radiometer": "radiometer_basic" in selected_ids,
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates(["budget", "sensor_ids"])
    frame = frame.sort_values(["budget", "steady_power", "cold_start_peak_power", "sensor_ids"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "physical_event_feasible_subsets.csv", index=False)
    return frame


def summarize(frame: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for budget, group in frame.groupby("budget", sort=True):
        rows.append(
            {
                "budget": float(budget),
                "feasible_subset_count": int(len(group)),
                "laser_subset_count": int(group["has_laser"].sum()),
                "laser_fc4_subset_count": int((group["has_laser"] & group["has_fc4"]).sum()),
                "snow_particle_fc4_subset_count": int(
                    (group["has_snow_particle_counter"] & group["has_fc4"]).sum()
                ),
                "min_power": float(group["steady_power"].min()) if not group.empty else float("nan"),
                "max_power": float(group["steady_power"].max()) if not group.empty else float("nan"),
                "lowest_power_with_laser": float(group[group["has_laser"]]["steady_power"].min())
                if bool(group["has_laser"].any())
                else float("nan"),
                "lowest_power_with_laser_fc4": float(group[group["has_laser"] & group["has_fc4"]]["steady_power"].min())
                if bool((group["has_laser"] & group["has_fc4"]).any())
                else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "physical_event_feasibility_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight feasible subsets for the V3.1 physical-event scenario.")
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v2.yaml")
    parser.add_argument("--budgets", nargs="+", type=float, default=[1.00, 1.10, 1.20])
    parser.add_argument("--startup-peak-budget", type=float, default=1.60)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--required-sensors", nargs="*", default=["met_station_core"])
    parser.add_argument("--out-dir", default="reports/physical_event_v2_preflight")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frame = enumerate_feasible_subsets(
        sensor_cfg=resolve_sensor_cfg(str(args.sensor_cfg)),
        budgets=[float(x) for x in args.budgets],
        startup_peak_budget=float(args.startup_peak_budget),
        max_active=int(args.max_active) if args.max_active is not None else None,
        required_sensors=tuple(str(x) for x in args.required_sensors),
        out_dir=out_dir,
    )
    summary = summarize(frame, out_dir)
    print(summary.to_string(index=False))
    print(out_dir / "physical_event_feasible_subsets.csv")


if __name__ == "__main__":
    main()
