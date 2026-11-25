#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(value)


def load_string_array(payload: Any, key: str) -> list[str]:
    if key not in payload:
        return []
    values = np.asarray(payload[key]).reshape(-1)
    return [str(value) for value in values.tolist()]


def mask_keys(selected: np.ndarray) -> list[str]:
    masks = np.asarray(selected, dtype=int)
    if masks.ndim != 2:
        raise ValueError(f"selected_masks must be 2D, got {masks.shape}")
    return ["".join("1" if int(x) else "0" for x in row) for row in masks]


def entropy_bits(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return float("nan")
    p = values[values > 0] / total
    return float(-np.sum(p * np.log2(p)))


def categorical_entropy(keys: list[str]) -> float:
    if not keys:
        return float("nan")
    _, counts = np.unique(np.asarray(keys, dtype=object), return_counts=True)
    return entropy_bits(counts)


def categorical_mutual_information_bits(keys: list[str], labels: np.ndarray) -> float:
    if not keys:
        return float("nan")
    label = np.asarray(labels).reshape(-1)
    if label.size != len(keys) or label.size == 0:
        return float("nan")
    if np.all(label == label[0]):
        return 0.0
    h_mask = categorical_entropy(keys)
    h_cond = 0.0
    for value in np.unique(label):
        idx = np.flatnonzero(label == value)
        if idx.size == 0:
            continue
        sub_keys = [keys[int(i)] for i in idx]
        h_cond += float(idx.size) / float(label.size) * categorical_entropy(sub_keys)
    return float(max(0.0, h_mask - h_cond))


def event_mask_mutual_information_bits(keys: list[str], events: np.ndarray) -> float:
    event = np.asarray(events, dtype=bool).reshape(-1)
    return categorical_mutual_information_bits(keys, event)


def subtype_labels_from_payload(payload: Any, steps: int) -> tuple[np.ndarray | None, dict[str, int]]:
    if "truth" not in payload or "state_columns" not in payload:
        return None, {}
    truth = np.asarray(payload["truth"])
    if truth.ndim != 2 or truth.shape[0] != steps:
        return None, {}
    state_columns = [str(value) for value in np.asarray(payload["state_columns"]).reshape(-1).tolist()]
    subtype_names = [
        "event_subtype_particle_latent",
        "event_subtype_flux_latent",
        "event_subtype_thermal_latent",
    ]
    indices: list[int] = []
    for name in subtype_names:
        if name not in state_columns:
            return None, {}
        indices.append(state_columns.index(name))
    latent = np.asarray(truth[:, indices], dtype=float)
    labels = np.zeros(steps, dtype=int)
    active = np.isfinite(latent).all(axis=1) & (np.nanmax(latent, axis=1) > 0.0)
    if np.any(active):
        labels[active] = np.argmax(latent[active], axis=1) + 1
    counts = {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    return labels, counts


def label_sensor_l1(selected: np.ndarray, labels: np.ndarray | None) -> float:
    if labels is None:
        return float("nan")
    masks = np.asarray(selected, dtype=float)
    label = np.asarray(labels).reshape(-1)
    if label.size != masks.shape[0] or label.size == 0:
        return float("nan")
    means: list[np.ndarray] = []
    for value in np.unique(label):
        idx = np.flatnonzero(label == value)
        if idx.size:
            means.append(np.mean(masks[idx], axis=0))
    if len(means) < 2:
        return 0.0
    best = 0.0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            best = max(best, float(np.abs(means[i] - means[j]).sum()))
    return float(best)


def best_period_match(selected: np.ndarray, max_period: int) -> tuple[int, float]:
    masks = np.asarray(selected, dtype=int)
    n_steps = int(masks.shape[0])
    if n_steps <= 2:
        return 0, float("nan")
    limit = min(int(max_period), max(1, n_steps // 2))
    best_p = 1
    best_score = -1.0
    for period in range(1, limit + 1):
        same = np.all(masks[period:] == masks[:-period], axis=1)
        score = float(np.mean(same)) if same.size else float("nan")
        if np.isfinite(score) and score > best_score:
            best_p = int(period)
            best_score = float(score)
    return best_p, best_score


def transition_keys(keys: list[str]) -> list[str]:
    return [f"{a}->{b}" for a, b in zip(keys[:-1], keys[1:])]


def topk_fraction(keys: list[str], k: int) -> float:
    if not keys:
        return float("nan")
    _, counts = np.unique(np.asarray(keys, dtype=object), return_counts=True)
    counts = np.sort(counts)[::-1]
    return float(np.sum(counts[: int(k)]) / len(keys))


def sensor_duty(selected: np.ndarray, event_flags: np.ndarray, value: bool | None) -> np.ndarray:
    masks = np.asarray(selected, dtype=float)
    if value is None:
        idx = np.arange(masks.shape[0])
    else:
        events = np.asarray(event_flags, dtype=bool).reshape(-1)
        idx = np.flatnonzero(events == bool(value))
    if idx.size == 0:
        return np.full(masks.shape[1], np.nan, dtype=float)
    return np.mean(masks[idx], axis=0)


def audit_rollout(
    path: Path,
    *,
    max_period: int,
    fixed_top1_threshold: float,
    simple_top3_threshold: float,
    simple_period_threshold: float,
    min_unique_masks: int,
    min_mask_entropy_bits: float,
    min_transition_entropy_bits: float,
    min_event_sensor_l1: float,
    min_event_mi_bits: float,
    min_subtype_sensor_l1: float,
    min_subtype_mi_bits: float,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        selected = np.asarray(payload["selected_masks"], dtype=int)
        events = np.asarray(payload["event_flags"], dtype=bool).reshape(-1)
        sensor_ids = load_string_array(payload, "sensor_ids")
        policy_values = load_string_array(payload, "policy")
        subtype_labels, subtype_label_counts = subtype_labels_from_payload(payload, int(selected.shape[0]))

    if selected.ndim != 2 or selected.shape[0] == 0:
        raise ValueError(f"{path}: empty or invalid selected_masks shape {selected.shape}")
    if events.size != selected.shape[0]:
        events = np.resize(events, selected.shape[0]).astype(bool)

    keys = mask_keys(selected)
    unique_keys, counts = np.unique(np.asarray(keys, dtype=object), return_counts=True)
    transition = transition_keys(keys)
    period, period_score = best_period_match(selected, max_period=max_period)

    all_duty = sensor_duty(selected, events, None)
    event_duty = sensor_duty(selected, events, True)
    non_event_duty = sensor_duty(selected, events, False)
    event_delta = np.abs(event_duty - non_event_duty)
    finite_delta = event_delta[np.isfinite(event_delta)]
    sensor_l1 = float(np.nansum(event_delta))
    sensor_linf = float(np.nanmax(event_delta)) if finite_delta.size else float("nan")

    top1 = topk_fraction(keys, 1)
    top2 = topk_fraction(keys, 2)
    top3 = topk_fraction(keys, 3)
    mask_entropy = entropy_bits(counts)
    transition_entropy = categorical_entropy(transition)
    event_mi = event_mask_mutual_information_bits(keys, events)
    subtype_mi = (
        categorical_mutual_information_bits(keys, subtype_labels)
        if subtype_labels is not None
        else float("nan")
    )
    subtype_sensor_l1 = label_sensor_l1(selected, subtype_labels)
    switch_rate = (
        float(np.mean(np.mean(np.abs(np.diff(selected.astype(float), axis=0)), axis=1)))
        if selected.shape[0] > 1
        else 0.0
    )

    event_state_dependent = bool(
        (np.isfinite(sensor_l1) and sensor_l1 >= float(min_event_sensor_l1))
        or (np.isfinite(event_mi) and event_mi >= float(min_event_mi_bits))
    )
    subtype_state_dependent = bool(
        (np.isfinite(subtype_sensor_l1) and subtype_sensor_l1 >= float(min_subtype_sensor_l1))
        or (np.isfinite(subtype_mi) and subtype_mi >= float(min_subtype_mi_bits))
    )
    state_dependent = bool(event_state_dependent or subtype_state_dependent)
    fixed_like = bool(
        len(unique_keys) <= 1
        or (np.isfinite(top1) and top1 >= float(fixed_top1_threshold))
        or (len(unique_keys) < int(min_unique_masks) and not state_dependent)
    )
    simple_cycle_like = bool(
        np.isfinite(top3)
        and np.isfinite(period_score)
        and int(period) > 1
        and top3 >= float(simple_top3_threshold)
        and period_score >= float(simple_period_threshold)
        and not state_dependent
    )
    low_complexity = bool(
        (np.isfinite(mask_entropy) and mask_entropy < float(min_mask_entropy_bits))
        or (
            np.isfinite(transition_entropy)
            and transition_entropy < float(min_transition_entropy_bits)
        )
    )
    weak_state_dependence = bool(
        not event_state_dependent
        and not subtype_state_dependent
    )
    gate_pass = bool(
        not fixed_like
        and not simple_cycle_like
        and not low_complexity
        and not weak_state_dependence
    )

    top_indices = np.argsort(counts)[::-1][:5]
    top_masks = [
        {
            "mask": str(unique_keys[int(idx)]),
            "count": int(counts[int(idx)]),
            "fraction": float(counts[int(idx)] / selected.shape[0]),
        }
        for idx in top_indices
    ]
    sensor_rows = []
    for idx in range(selected.shape[1]):
        sensor_rows.append(
            {
                "sensor": sensor_ids[idx] if idx < len(sensor_ids) else f"sensor_{idx}",
                "duty": float(all_duty[idx]),
                "event_duty": float(event_duty[idx]) if np.isfinite(event_duty[idx]) else float("nan"),
                "non_event_duty": float(non_event_duty[idx]) if np.isfinite(non_event_duty[idx]) else float("nan"),
                "event_abs_delta": float(event_delta[idx]) if np.isfinite(event_delta[idx]) else float("nan"),
            }
        )

    return {
        "path": str(path),
        "policy": policy_values[0] if policy_values else path.stem.removeprefix("rollout_"),
        "steps": int(selected.shape[0]),
        "sensors": int(selected.shape[1]),
        "event_rate": float(np.mean(events)) if events.size else float("nan"),
        "unique_mask_count": int(len(unique_keys)),
        "unique_mask_fraction": float(len(unique_keys) / selected.shape[0]),
        "top1_mask_fraction": float(top1),
        "top2_mask_fraction": float(top2),
        "top3_mask_fraction": float(top3),
        "mask_entropy_bits": float(mask_entropy),
        "transition_count": int(len(set(transition))),
        "transition_entropy_bits": float(transition_entropy),
        "switches_per_step": float(switch_rate),
        "best_period": int(period),
        "best_period_match": float(period_score),
        "event_mask_mi_bits": float(event_mi),
        "event_sensor_l1": float(sensor_l1),
        "event_sensor_linf": float(sensor_linf),
        "event_state_dependent": event_state_dependent,
        "subtype_mask_mi_bits": float(subtype_mi),
        "subtype_sensor_l1": float(subtype_sensor_l1),
        "subtype_state_dependent": subtype_state_dependent,
        "subtype_label_counts_json": json.dumps(subtype_label_counts, ensure_ascii=False),
        "state_dependent": state_dependent,
        "fixed_like": fixed_like,
        "simple_cycle_like": simple_cycle_like,
        "low_complexity": low_complexity,
        "weak_state_dependence": weak_state_dependence,
        "behavior_complexity_gate_pass": gate_pass,
        "top_masks_json": json.dumps(top_masks, ensure_ascii=False),
        "sensor_duty_json": json.dumps(sensor_rows, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit rollout scheduling complexity to reject fixed subsets and "
            "simple periodic sensor-combination cycles."
        )
    )
    parser.add_argument("rollouts", nargs="+", help="Rollout NPZ files.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory.")
    parser.add_argument("--out-name", default="behavior_complexity_summary.csv")
    parser.add_argument("--max-period", type=int, default=64)
    parser.add_argument("--fixed-top1-threshold", type=float, default=0.95)
    parser.add_argument("--simple-top3-threshold", type=float, default=0.85)
    parser.add_argument("--simple-period-threshold", type=float, default=0.90)
    parser.add_argument("--min-unique-masks", type=int, default=5)
    parser.add_argument("--min-mask-entropy-bits", type=float, default=1.50)
    parser.add_argument("--min-transition-entropy-bits", type=float, default=1.25)
    parser.add_argument("--min-event-sensor-l1", type=float, default=0.50)
    parser.add_argument("--min-event-mi-bits", type=float, default=0.10)
    parser.add_argument("--min-subtype-sensor-l1", type=float, default=1.00)
    parser.add_argument("--min-subtype-mi-bits", type=float, default=0.25)
    args = parser.parse_args()

    rows = [
        audit_rollout(
            resolve_path(path),
            max_period=int(args.max_period),
            fixed_top1_threshold=float(args.fixed_top1_threshold),
            simple_top3_threshold=float(args.simple_top3_threshold),
            simple_period_threshold=float(args.simple_period_threshold),
            min_unique_masks=int(args.min_unique_masks),
            min_mask_entropy_bits=float(args.min_mask_entropy_bits),
            min_transition_entropy_bits=float(args.min_transition_entropy_bits),
            min_event_sensor_l1=float(args.min_event_sensor_l1),
            min_event_mi_bits=float(args.min_event_mi_bits),
            min_subtype_sensor_l1=float(args.min_subtype_sensor_l1),
            min_subtype_mi_bits=float(args.min_subtype_mi_bits),
        )
        for path in args.rollouts
    ]

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / str(args.out_name)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (out_path.with_suffix(".json")).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path)

    for row in rows:
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "policy",
                        "steps",
                        "unique_mask_count",
                        "top3_mask_fraction",
                        "mask_entropy_bits",
                        "transition_entropy_bits",
                        "best_period",
                        "best_period_match",
                        "event_sensor_l1",
                        "event_mask_mi_bits",
                        "event_state_dependent",
                        "subtype_sensor_l1",
                        "subtype_mask_mi_bits",
                        "subtype_state_dependent",
                        "state_dependent",
                        "fixed_like",
                        "simple_cycle_like",
                        "low_complexity",
                        "weak_state_dependence",
                        "behavior_complexity_gate_pass",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
