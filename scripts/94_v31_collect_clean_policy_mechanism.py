#!/usr/bin/env python
"""Collect offline mechanism evidence from frozen clean-policy rollouts."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBTYPE_NAMES = {0: "calm", 1: "particle", 2: "flux", 3: "thermal"}


def load_behavior_module() -> Any:
    path = Path(__file__).with_name("71_v31_behavior_complexity_audit.py")
    spec = importlib.util.spec_from_file_location("v31_behavior_complexity", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_seed(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot parse seed from {path}")
    return int(match.group(1))


def string_values(payload: Any, key: str) -> list[str]:
    return [str(value) for value in np.asarray(payload[key]).reshape(-1).tolist()]


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, draws: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(draws, data.size))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def selected_static_description(metadata: dict[str, Any]) -> tuple[int, str]:
    selected = dict(metadata.get("selected_static_reference", {}))
    action_idx = int(selected.get("action_idx", -1))
    sensors = selected.get("sensor_ids", selected.get("sensors", ()))
    if isinstance(sensors, str):
        sensor_text = sensors
    else:
        sensor_text = "|".join(str(value) for value in sensors)
    return action_idx, sensor_text


def collect_rollout(
    path: Path,
    *,
    behavior_module: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    seed = parse_seed(path)
    run_dir = path.parent
    metadata = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    policy_cfg = dict(metadata.get("custom_ppo", {}))
    alert_cfg = dict(metadata.get("agent_alert_context", {}))
    if bool(policy_cfg.get("subtype_router_enabled", False)):
        raise ValueError(f"Hard-router rollout is not clean evidence: {path}")
    if bool(alert_cfg.get("include_event_flag_in_state", True)):
        raise ValueError(f"Exact event flag appears in policy state: {path}")
    if bool(alert_cfg.get("truth_event_labels_used_online", True)):
        raise ValueError(f"Metadata records online truth-label use: {path}")

    with np.load(path, allow_pickle=True) as payload:
        selected = np.asarray(payload["selected_masks"], dtype=bool)
        sensor_ids = string_values(payload, "sensor_ids")
        subtype_labels, subtype_counts = behavior_module.subtype_labels_from_payload(
            payload,
            int(selected.shape[0]),
        )
    if subtype_labels is None:
        raise ValueError(f"Offline subtype labels unavailable in {path}")
    if selected.ndim != 2 or selected.shape[1] != len(sensor_ids):
        raise ValueError(f"Invalid selected-mask shape in {path}: {selected.shape}")

    audit = behavior_module.audit_rollout(
        path,
        max_period=64,
        fixed_top1_threshold=0.95,
        simple_top3_threshold=0.85,
        simple_period_threshold=0.90,
        min_unique_masks=5,
        min_mask_entropy_bits=1.50,
        min_transition_entropy_bits=1.25,
        min_event_sensor_l1=0.50,
        min_event_mi_bits=0.10,
        min_subtype_sensor_l1=1.00,
        min_subtype_mi_bits=0.25,
    )
    selected_action, selected_sensors = selected_static_description(metadata)
    behavior_row: dict[str, Any] = {
        "seed": seed,
        "run_dir": str(run_dir),
        "selected_static_action_idx": selected_action,
        "selected_static_sensors": selected_sensors,
        "subtype_counts_json": json.dumps(subtype_counts, sort_keys=True),
    }
    for key in (
        "steps",
        "unique_mask_count",
        "top1_mask_fraction",
        "top3_mask_fraction",
        "mask_entropy_bits",
        "transition_entropy_bits",
        "switches_per_step",
        "best_period",
        "best_period_match",
        "event_mask_mi_bits",
        "event_sensor_l1",
        "subtype_mask_mi_bits",
        "subtype_sensor_l1",
        "state_dependent",
        "fixed_like",
        "simple_cycle_like",
        "behavior_complexity_gate_pass",
    ):
        behavior_row[key] = audit[key]

    subtype_rows: list[dict[str, Any]] = []
    for subtype_id, subtype_name in SUBTYPE_NAMES.items():
        idx = np.flatnonzero(np.asarray(subtype_labels, dtype=int) == int(subtype_id))
        if idx.size == 0:
            continue
        duty = np.mean(selected[idx].astype(float), axis=0)
        for sensor_idx, sensor_id in enumerate(sensor_ids):
            subtype_rows.append(
                {
                    "seed": seed,
                    "subtype_id": subtype_id,
                    "subtype": subtype_name,
                    "n_steps": int(idx.size),
                    "sensor": sensor_id,
                    "selection_fraction": float(duty[sensor_idx]),
                }
            )

    sensor_rows = [
        {
            "seed": seed,
            "sensor": sensor_id,
            "selection_fraction": float(np.mean(selected[:, sensor_idx])),
        }
        for sensor_idx, sensor_id in enumerate(sensor_ids)
    ]
    return behavior_row, subtype_rows, sensor_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-glob", required=True)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()

    wanted = set(args.seeds or ())
    paths = [Path(value) for value in sorted(glob.glob(args.rollout_glob))]
    if wanted:
        paths = [path for path in paths if parse_seed(path) in wanted]
    if not paths:
        raise SystemExit("No rollout paths matched")

    behavior_module = load_behavior_module()
    behavior_rows: list[dict[str, Any]] = []
    subtype_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    for path in paths:
        behavior, subtype, sensors = collect_rollout(path, behavior_module=behavior_module)
        behavior_rows.append(behavior)
        subtype_rows.extend(subtype)
        sensor_rows.extend(sensors)

    behavior_table = pd.DataFrame(behavior_rows).sort_values("seed").reset_index(drop=True)
    subtype_table = pd.DataFrame(subtype_rows).sort_values(["subtype_id", "sensor", "seed"])
    sensor_table = pd.DataFrame(sensor_rows).sort_values(["sensor", "seed"])
    subtype_summary = (
        subtype_table.groupby(["subtype_id", "subtype", "sensor"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            selection_fraction_mean=("selection_fraction", "mean"),
            selection_fraction_std=("selection_fraction", "std"),
        )
        .sort_values(["subtype_id", "sensor"])
    )
    sensor_summary = (
        sensor_table.groupby("sensor", as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            selection_fraction_mean=("selection_fraction", "mean"),
            selection_fraction_min=("selection_fraction", "min"),
            selection_fraction_max=("selection_fraction", "max"),
        )
        .sort_values("sensor")
    )

    summary: dict[str, Any] = {
        "seed_count": int(behavior_table["seed"].nunique()),
        "seeds": [int(value) for value in behavior_table["seed"].tolist()],
        "behavior_gate_passes": int(behavior_table["behavior_complexity_gate_pass"].sum()),
        "fixed_like_count": int(behavior_table["fixed_like"].sum()),
        "simple_cycle_like_count": int(behavior_table["simple_cycle_like"].sum()),
        "offline_subtype_labels_used_only_for_grouping": True,
    }
    for index, column in enumerate(
        ("mask_entropy_bits", "transition_entropy_bits", "subtype_mask_mi_bits", "switches_per_step")
    ):
        values = behavior_table[column].to_numpy(dtype=float)
        ci = bootstrap_mean_ci(values, seed=94_000 + index, draws=int(args.bootstrap_draws))
        summary[column] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "bootstrap_95_ci": [ci[0], ci[1]],
        }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_table.to_csv(out_dir / "clean_policy_behavior_seed_metrics.csv", index=False)
    subtype_table.to_csv(out_dir / "clean_policy_subtype_sensor_duty.csv", index=False)
    subtype_summary.to_csv(out_dir / "clean_policy_subtype_sensor_duty_summary.csv", index=False)
    sensor_table.to_csv(out_dir / "clean_policy_sensor_duty.csv", index=False)
    sensor_summary.to_csv(out_dir / "clean_policy_sensor_duty_summary.csv", index=False)
    (out_dir / "clean_policy_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
