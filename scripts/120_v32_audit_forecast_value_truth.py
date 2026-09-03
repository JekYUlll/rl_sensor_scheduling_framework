#!/usr/bin/env python3
"""Audit frozen Stage-P truth before any forecaster or policy is fitted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FACTORS = ("flux", "particle", "thermal")
PARTITIONS = {
    "forecaster_training": (0, 12600),
    "policy_training": (12600, 30600),
    "calibration_validation": (30600, 33300),
    "test": (33300, 36000),
}


def binary_run_lengths(values: np.ndarray, level: bool) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in values:
        if bool(value) == bool(level):
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def audit(path: Path, lead: int, activity_aligned_transport_demand: bool = False) -> dict[str, object]:
    frame = pd.read_csv(path)
    state_columns = [f"generator_{factor}_demand_state" for factor in FACTORS]
    forecast_columns = [f"agent_context_forecast_{factor}_demand" for factor in FACTORS]
    required = [*state_columns, *forecast_columns, "blowing_snow_active",
                "air_temperature_c", "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s", "snow_particle_mean_velocity_ms",
                "snow_particle_mean_diameter_mm"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"missing Stage-P columns in {path}: {missing}")
    train = frame.iloc[slice(*PARTITIONS["policy_training"])]
    train_active = train["blowing_snow_active"].to_numpy(dtype=bool)
    medians = {}
    for factor, column in zip(FACTORS, state_columns, strict=True):
        values = train[column].to_numpy(dtype=float)
        threshold_values = values[train_active] if activity_aligned_transport_demand and factor in {"flux", "particle"} else values
        if len(threshold_values) == 0:
            raise ValueError(f"no active policy-training samples for {factor} in {path}")
        medians[column] = float(np.median(threshold_values))
    quantiles = {
        column: tuple(float(value) for value in np.quantile(
            train[column].to_numpy(dtype=float)[train_active]
            if activity_aligned_transport_demand and factor in {"flux", "particle"}
            else train[column].to_numpy(dtype=float),
            [0.25, 0.75],
        ))
        for factor, column in zip(FACTORS, state_columns, strict=True)
    }

    partition_rows: dict[str, object] = {}
    support_pass = True
    persistence_pass = True
    for partition, bounds in PARTITIONS.items():
        part = frame.iloc[slice(*bounds)]
        part_active = part["blowing_snow_active"].to_numpy(dtype=bool)
        bits = np.column_stack([
            part[column].to_numpy(dtype=float) >= medians[column]
            for column in state_columns
        ])
        codes = bits[:, 0].astype(int) * 4 + bits[:, 1].astype(int) * 2 + bits[:, 2].astype(int)
        support_domain = part_active if activity_aligned_transport_demand else np.ones(len(part), dtype=bool)
        support = {
            f"{code:03b}": float(np.mean(codes[support_domain] == code))
            if np.any(support_domain) else 0.0
            for code in range(8)
        }
        if partition in {"calibration_validation", "test"}:
            support_pass &= all(value >= 0.05 for value in support.values())
            if activity_aligned_transport_demand:
                support_pass &= float(np.mean(part_active)) >= 0.20
        runs: dict[str, object] = {}
        for factor, column in zip(FACTORS, state_columns, strict=True):
            high = part[column].to_numpy(dtype=float) >= medians[column]
            factor_runs = {}
            for label, level in (("low", False), ("high", True)):
                lengths = binary_run_lengths(high, level)
                median = float(np.median(lengths)) if lengths else 0.0
                factor_runs[label] = {"runs": len(lengths), "median_steps": median}
                if partition in {"calibration_validation", "test"}:
                    persistence_pass &= median >= 6.0
            runs[factor] = factor_runs
        partition_rows[partition] = {
            "bounds": list(bounds),
            "activity_fraction": float(np.mean(part_active)),
            "support_domain": "active_samples" if activity_aligned_transport_demand else "all_samples",
            "state_support": support,
            "runs": runs,
        }

    test_start, test_end = PARTITIONS["test"]
    forecast_correlations = {}
    forecast_pass = True
    for factor, state_column, forecast_column in zip(FACTORS, state_columns, forecast_columns, strict=True):
        left = frame[forecast_column].to_numpy(dtype=float)[test_start:test_end - lead]
        right = frame[state_column].to_numpy(dtype=float)[test_start + lead:test_end]
        correlation = safe_corr(left, right)
        forecast_correlations[factor] = correlation
        forecast_pass &= bool(np.isfinite(correlation) and correlation >= 0.55)

    active = frame["blowing_snow_active"].to_numpy(dtype=bool)
    flux_state = frame[state_columns[0]].to_numpy(dtype=float)
    particle_state = frame[state_columns[1]].to_numpy(dtype=float)
    thermal_state = frame[state_columns[2]].to_numpy(dtype=float)
    base_flux = 1.0e-6 + 4.5e-5 * np.square(flux_state)
    residuals = {
        "flux": np.log(np.maximum(frame["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float), 1.0e-12) / base_flux),
        "particle_velocity": frame["snow_particle_mean_velocity_ms"].to_numpy(dtype=float) - (1.5 + 7.0 * particle_state),
        "particle_diameter": frame["snow_particle_mean_diameter_mm"].to_numpy(dtype=float) - (0.07 + 0.22 * particle_state + 0.04 * (1.0 - thermal_state)),
        "thermal": frame["snow_surface_temperature_c"].to_numpy(dtype=float) - (frame["air_temperature_c"].to_numpy(dtype=float) - (1.0 + 5.0 * thermal_state)),
    }
    residual_specs = {
        "flux": (state_columns[0], True),
        "particle_velocity": (state_columns[1], True),
        "particle_diameter": (state_columns[1], True),
        "thermal": (state_columns[2], False),
    }
    residual_ratios = {}
    residual_pass = True
    for name, values in residuals.items():
        state_column, event_only = residual_specs[name]
        low_q, high_q = quantiles[state_column]
        states = frame[state_column].to_numpy(dtype=float)
        interval = np.zeros(len(frame), dtype=bool)
        interval[test_start:test_end] = True
        if event_only:
            interval &= active
        low_values = values[interval & (states <= low_q)]
        high_values = values[interval & (states >= high_q)]
        low_std = float(np.std(low_values)) if len(low_values) else float("nan")
        high_std = float(np.std(high_values)) if len(high_values) else float("nan")
        ratio = high_std / max(low_std, 1.0e-12) if np.isfinite(high_std) and np.isfinite(low_std) else float("nan")
        residual_ratios[name] = {
            "low_samples": int(len(low_values)), "high_samples": int(len(high_values)),
            "low_std": low_std, "high_std": high_std, "ratio": float(ratio),
        }
        residual_pass &= bool(len(low_values) >= 20 and len(high_values) >= 20 and ratio >= 1.5)

    finite_columns = [
        "snow_mass_flux_kg_m2_s", "snow_particle_mean_velocity_ms",
        "snow_particle_mean_diameter_mm", "snow_surface_temperature_c",
        *state_columns, *forecast_columns,
    ]
    finite_pass = bool(np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all())
    bounds_pass = bool(
        frame["snow_mass_flux_kg_m2_s"].ge(0.0).all()
        and frame.loc[active, "snow_particle_mean_velocity_ms"].between(0.0, 20.0).all()
        and frame.loc[active, "snow_particle_mean_diameter_mm"].between(0.04, 0.5).all()
        and frame["snow_surface_temperature_c"].between(-80.0, 10.0).all()
    )
    gates = {
        "partition_support": bool(support_pass),
        "six_step_persistence": bool(persistence_pass),
        "forecast_correlation": bool(forecast_pass),
        "residual_scale_ratio": bool(residual_pass),
        "finite_physical_bounds": bool(finite_pass and bounds_pass),
    }
    return {
        "truth_csv": str(path), "lead_steps": int(lead),
        "activity_aligned_transport_demand": bool(activity_aligned_transport_demand),
        "thresholds": medians,
        "partitions": partition_rows, "forecast_correlations": forecast_correlations,
        "residual_scale_ratios": residual_ratios, "gates": gates,
        "all_truth_gates_pass": bool(all(gates.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-csv", action="append", type=Path, required=True)
    parser.add_argument("--lead-steps", type=int, default=8)
    parser.add_argument("--activity-aligned-transport-demand", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = [audit(
        path,
        int(args.lead_steps),
        activity_aligned_transport_demand=bool(args.activity_aligned_transport_demand),
    ) for path in args.truth_csv]
    payload = {"reports": reports, "all_seeds_pass": all(report["all_truth_gates_pass"] for report in reports)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
