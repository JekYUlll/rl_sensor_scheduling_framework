#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


POLICY_ORDER = (
    "full_open_unconstrained",
    "oracle_static_projected",
    "validation_selected_static",
    "custom_ppo",
    "feasible_static_projected",
    "round_robin",
    "aoi",
    "random",
)


def _budget_from_name(name: str) -> float:
    match = re.search(r"budget(?P<value>\d+p\d+)", name)
    if not match:
        match = re.search(r"budget(?P<value>\d+(?:\.\d+)?)", name)
    if not match:
        raise ValueError(f"Cannot parse budget from {name!r}")
    return float(match.group("value").replace("p", "."))


def _seed_from_name(name: str) -> int:
    match = re.search(r"seed(?P<seed>\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse seed from {name!r}")
    return int(match.group("seed"))


def _budget_for_run_dir(run_dir: Path) -> float:
    try:
        return _budget_from_name(run_dir.name)
    except ValueError:
        metadata_path = run_dir / "v2_ppo_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            constraints = metadata.get("constraints", {})
            if "per_step_budget" in constraints:
                return float(constraints["per_step_budget"])
        raise


def _run_dirs(report_dir: Path) -> list[Path]:
    if any(report_dir.glob("rollout_*.npz")):
        return [report_dir]
    raw_dir = report_dir / "raw"
    if raw_dir.exists():
        return sorted(path for path in raw_dir.glob("budget*_seed*") if path.is_dir())
    return sorted(path for path in report_dir.glob("*budget*_seed*") if path.is_dir())


def _bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def _load_sensor_specs(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data.get("sensors", []))


def _subset_names(sensors: list[dict[str, object]], indices: tuple[int, ...]) -> str:
    return "|".join(str(sensors[idx]["sensor_id"]) for idx in indices)


def _candidate_budget_rows(sensors: list[dict[str, object]], budgets: list[float], startup_peak_budget: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_sensors = len(sensors)
    for budget in budgets:
        feasible: list[tuple[float, float, tuple[int, ...]]] = []
        for size in range(1, n_sensors + 1):
            for indices in itertools.combinations(range(n_sensors), size):
                steady = float(sum(float(sensors[idx].get("power_cost", 0.0)) for idx in indices))
                peak = float(
                    sum(float(sensors[idx].get("startup_peak_power", sensors[idx].get("power_cost", 0.0))) for idx in indices)
                )
                if size <= 4 and steady <= float(budget) + 1e-12 and peak <= float(startup_peak_budget) + 1e-12:
                    feasible.append((steady, peak, indices))
        max_size = max((len(indices) for _, _, indices in feasible), default=0)
        top = sorted(feasible, key=lambda x: (-len(x[2]), x[0], x[1], x[2]))[:8]
        for rank, (steady, peak, indices) in enumerate(top, start=1):
            rows.append(
                {
                    "budget": float(budget),
                    "rank": int(rank),
                    "feasible_count": int(len(feasible)),
                    "max_feasible_size": int(max_size),
                    "steady_power": float(steady),
                    "startup_peak_power": float(peak),
                    "sensor_count": int(len(indices)),
                    "sensor_ids": _subset_names(sensors, indices),
                }
            )
    return rows


def _event_stats(run_dir: Path) -> dict[str, float | int]:
    truth_path = run_dir / "truth_v31.csv"
    if not truth_path.exists():
        metadata_path = run_dir / "v2_ppo_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate = Path(str(metadata.get("truth_csv", "")))
            if candidate.exists():
                truth_path = candidate
    if not truth_path.exists():
        return {}
    truth = pd.read_csv(truth_path)
    event = truth["event_flag"].astype(bool).to_numpy() if "event_flag" in truth else np.zeros(len(truth), dtype=bool)
    storm = truth["storm_flag"].astype(bool).to_numpy() if "storm_flag" in truth else np.zeros(len(truth), dtype=bool)
    active = (
        truth["blowing_snow_active"].astype(bool).to_numpy()
        if "blowing_snow_active" in truth
        else event
    )
    runs = _bool_runs(event)
    durations = np.asarray([end - start for start, end in runs], dtype=float)
    window = 512
    if event.size >= window:
        fractions = np.asarray([event[idx : idx + window].mean() for idx in range(0, event.size - window + 1, window)])
    else:
        fractions = np.asarray([], dtype=float)
    return {
        "truth_steps": int(len(truth)),
        "truth_event_rate": float(event.mean()) if event.size else float("nan"),
        "truth_storm_rate": float(storm.mean()) if storm.size else float("nan"),
        "truth_blowing_snow_active_rate": float(active.mean()) if active.size else float("nan"),
        "truth_event_run_count": int(len(runs)),
        "truth_event_duration_mean_steps": float(durations.mean()) if durations.size else 0.0,
        "truth_event_duration_median_steps": float(np.median(durations)) if durations.size else 0.0,
        "truth_event_fraction_512_mean": float(fractions.mean()) if fractions.size else float("nan"),
        "truth_event_fraction_512_max": float(fractions.max()) if fractions.size else float("nan"),
        "truth_event_fraction_512_gt_0p75": float(np.mean(fractions > 0.75)) if fractions.size else float("nan"),
    }


def _rollout_rows(run_dir: Path, budget: float, seed: int, policy: str) -> list[dict[str, object]]:
    path = run_dir / f"rollout_{policy}.npz"
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=True)
    selected = np.asarray(data["selected_masks"], dtype=bool)
    modes = np.asarray(data["mode_ids"], dtype=int)
    power = np.asarray(data["powers"], dtype=float) if "powers" in data.files else np.asarray(data["power"], dtype=float)
    peaks = np.asarray(data["peaks"], dtype=float) if "peaks" in data.files else np.asarray([], dtype=float)
    sensor_ids = [str(x) for x in data["sensor_ids"]]
    warmup_abort_count = int(np.asarray(data["warmup_abort_count"]).reshape(-1)[0]) if "warmup_abort_count" in data.files else -1
    event = np.asarray(data["event_flags"], dtype=bool) if "event_flags" in data.files else np.zeros(selected.shape[0], dtype=bool)
    prev = np.zeros_like(selected)
    prev[1:] = selected[:-1]
    switch_count = int(np.sum(selected != prev))
    rows: list[dict[str, object]] = [
        {
            "level": "policy",
            "budget": float(budget),
            "seed": int(seed),
            "policy": policy,
            "sensor": "",
            "steps": int(selected.shape[0]),
            "selected_rate": float(selected.mean()),
            "active_rate": float((modes == 2).mean()),
            "warming_rate": float((modes == 1).mean()),
            "off_rate": float((modes == 0).mean()),
            "near_constant_sensors": int(sum(float(np.mean(selected[:, idx])) <= 0.02 or float(np.mean(selected[:, idx])) >= 0.98 for idx in range(selected.shape[1]))),
            "const_active_sensors": int(sum(float(np.mean(selected[:, idx])) >= 0.98 for idx in range(selected.shape[1]))),
            "const_off_sensors": int(sum(float(np.mean(selected[:, idx])) <= 0.02 for idx in range(selected.shape[1]))),
            "switch_count": switch_count,
            "switches_per_step": float(switch_count) / max(int(selected.shape[0]), 1),
            "warmup_abort_count": int(warmup_abort_count),
            "warmup_abort_rate": float(warmup_abort_count) / max(int(selected.shape[0]), 1) if warmup_abort_count >= 0 else float("nan"),
            "power_mean": float(power.mean()) if power.size else float("nan"),
            "power_max": float(power.max()) if power.size else float("nan"),
            "peak_max": float(peaks.max()) if peaks.size else float("nan"),
            "event_rate_rollout": float(event.mean()) if event.size else float("nan"),
        }
    ]
    for idx, sensor_id in enumerate(sensor_ids):
        sel = selected[:, idx]
        active = modes[:, idx] == 2
        warming = modes[:, idx] == 1
        event_sel = sel[event] if event.any() else np.asarray([], dtype=bool)
        nonevent_sel = sel[~event] if (~event).any() else np.asarray([], dtype=bool)
        rows.append(
            {
                "level": "sensor",
                "budget": float(budget),
                "seed": int(seed),
                "policy": policy,
                "sensor": sensor_id,
                "steps": int(selected.shape[0]),
                "selected_rate": float(sel.mean()),
                "active_rate": float(active.mean()),
                "warming_rate": float(warming.mean()),
                "off_rate": float((modes[:, idx] == 0).mean()),
                "near_constant_sensors": int(float(sel.mean()) <= 0.02 or float(sel.mean()) >= 0.98),
                "const_active_sensors": int(float(sel.mean()) >= 0.98),
                "const_off_sensors": int(float(sel.mean()) <= 0.02),
                "switch_count": int(np.sum(sel != np.r_[False, sel[:-1]])),
                "switches_per_step": float(np.sum(sel != np.r_[False, sel[:-1]])) / max(int(sel.size), 1),
                "warmup_abort_count": int(warmup_abort_count),
                "warmup_abort_rate": float(warmup_abort_count) / max(int(selected.shape[0]), 1) if warmup_abort_count >= 0 else float("nan"),
                "power_mean": float(power.mean()) if power.size else float("nan"),
                "power_max": float(power.max()) if power.size else float("nan"),
                "peak_max": float(peaks.max()) if peaks.size else float("nan"),
                "event_rate_rollout": float(event.mean()) if event.size else float("nan"),
                "selected_rate_event": float(event_sel.mean()) if event_sel.size else float("nan"),
                "selected_rate_non_event": float(nonevent_sel.mean()) if nonevent_sel.size else float("nan"),
                "event_selection_lift": float(event_sel.mean() - nonevent_sel.mean()) if event_sel.size and nonevent_sel.size else float("nan"),
            }
        )
    return rows


def _summarize_numeric(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    rendered = df.copy()
    for col in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[col]):
            rendered[col] = rendered[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
        else:
            rendered[col] = rendered[col].map(lambda value: "" if pd.isna(value) else str(value))
    cols = [str(col) for col in rendered.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in rendered.columns) + " |")
    return "\n".join(lines)


def _write_markdown(
    out_path: Path,
    *,
    sensor_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    event_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    sensor_summary: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# V3.1 Split-Protocol Behavior Diagnostics")
    lines.append("")
    lines.append("## Sensor Configuration")
    lines.append(_markdown_table(sensor_df))
    lines.append("")
    lines.append("## Feasible Subset Capacity")
    lines.append(_markdown_table(budget_df[budget_df["rank"] <= 5]))
    lines.append("")
    lines.append("## Truth Event Statistics")
    lines.append(_markdown_table(event_summary))
    lines.append("")
    lines.append("## Policy Behavior Summary")
    keep = [
        "budget",
        "policy",
        "near_constant_sensors_mean",
        "const_active_sensors_mean",
        "const_off_sensors_mean",
        "switches_per_step_mean",
        "warmup_abort_rate_mean",
        "power_mean_mean",
        "event_rate_rollout_mean",
    ]
    lines.append(_markdown_table(policy_summary[[col for col in keep if col in policy_summary.columns]]))
    lines.append("")
    lines.append("## Event-Conditioned High-Latency Sensor Use")
    high = sensor_summary[sensor_summary["sensor"].isin(["snow_particle_counter", "laser_disdrometer", "fc4_flux"])].copy()
    keep_sensor = [
        "budget",
        "policy",
        "sensor",
        "selected_rate_mean",
        "selected_rate_event_mean",
        "selected_rate_non_event_mean",
        "event_selection_lift_mean",
        "active_rate_mean",
        "warming_rate_mean",
        "switches_per_step_mean",
    ]
    lines.append(_markdown_table(high[[col for col in keep_sensor if col in high.columns]]))
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose V3.1 S2 scheduling behavior.")
    parser.add_argument("--report-dir", default="rl_sensor_scheduling_framework/reports/v31_s2_main")
    parser.add_argument("--sensor-cfg", default="rl_sensor_scheduling_framework/configs/sensors/windblown_sensors_balanced.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        sensor_stem = Path(args.sensor_cfg).stem
        dirname = "behavior_diagnostics" if sensor_stem == "windblown_sensors_balanced" else f"behavior_diagnostics_{sensor_stem}"
        out_dir = report_dir / dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    sensors = _load_sensor_specs(Path(args.sensor_cfg))
    sensor_df = pd.DataFrame(
        [
            {
                "order": idx,
                "sensor_id": str(item["sensor_id"]),
                "power_cost": float(item.get("power_cost", 0.0)),
                "startup_peak_power": float(item.get("startup_peak_power", item.get("power_cost", 0.0))),
                "warmup_steps": int(item.get("warmup_steps", 0)),
                "variables": "|".join(str(x) for x in item.get("variables", [])),
            }
            for idx, item in enumerate(sensors)
        ]
    )
    run_dirs = _run_dirs(report_dir)
    budgets = sorted({_budget_for_run_dir(path) for path in run_dirs})
    budget_df = pd.DataFrame(_candidate_budget_rows(sensors, budgets + [1.50, 1.55, 1.60], float(args.startup_peak_budget)))

    event_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        budget = _budget_for_run_dir(run_dir)
        seed = _seed_from_name(run_dir.name)
        event_row = {"budget": budget, "seed": seed, **_event_stats(run_dir)}
        event_rows.append(event_row)
        for policy in POLICY_ORDER:
            rollout_rows.extend(_rollout_rows(run_dir, budget, seed, policy))

    event_df = pd.DataFrame(event_rows)
    rollout_df = pd.DataFrame(rollout_rows)
    policy_df = rollout_df[rollout_df["level"] == "policy"].copy()
    sensor_rollout_df = rollout_df[rollout_df["level"] == "sensor"].copy()

    event_summary = _summarize_numeric(
        event_df,
        ["budget"],
        [
            "truth_event_rate",
            "truth_storm_rate",
            "truth_blowing_snow_active_rate",
            "truth_event_run_count",
            "truth_event_duration_mean_steps",
            "truth_event_duration_median_steps",
            "truth_event_fraction_512_mean",
            "truth_event_fraction_512_max",
            "truth_event_fraction_512_gt_0p75",
        ],
    )
    policy_summary = _summarize_numeric(
        policy_df,
        ["budget", "policy"],
        [
            "near_constant_sensors",
            "const_active_sensors",
            "const_off_sensors",
            "switches_per_step",
            "warmup_abort_rate",
            "power_mean",
            "power_max",
            "peak_max",
            "event_rate_rollout",
        ],
    )
    sensor_summary = _summarize_numeric(
        sensor_rollout_df,
        ["budget", "policy", "sensor"],
        [
            "selected_rate",
            "active_rate",
            "warming_rate",
            "switches_per_step",
            "selected_rate_event",
            "selected_rate_non_event",
            "event_selection_lift",
        ],
    )

    sensor_df.to_csv(out_dir / "sensor_config.csv", index=False)
    budget_df.to_csv(out_dir / "feasible_subset_capacity.csv", index=False)
    event_df.to_csv(out_dir / "truth_event_stats_long.csv", index=False)
    event_summary.to_csv(out_dir / "truth_event_stats_summary.csv", index=False)
    policy_df.to_csv(out_dir / "policy_behavior_long.csv", index=False)
    policy_summary.to_csv(out_dir / "policy_behavior_summary.csv", index=False)
    sensor_rollout_df.to_csv(out_dir / "sensor_behavior_long.csv", index=False)
    sensor_summary.to_csv(out_dir / "sensor_behavior_summary.csv", index=False)
    _write_markdown(
        out_dir / "diagnostic_summary.md",
        sensor_df=sensor_df,
        budget_df=budget_df,
        event_summary=event_summary,
        policy_summary=policy_summary,
        sensor_summary=sensor_summary,
    )
    print(out_dir)
    print((out_dir / "diagnostic_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
