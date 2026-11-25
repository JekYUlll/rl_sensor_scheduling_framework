#!/usr/bin/env python3
"""Add a frozen, causal cold-risk wind innovation to empirical-cold truth.

The perturbation represents persistent local wind uncertainty observed after a
cold availability episode.  It is generated from the decision-time cold-risk
proxy and is delayed before entering the target, so the scheduler cannot use a
future target or an event label.  This is a truth-only screen; no forecaster
or policy is trained here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build(frame: pd.DataFrame, seed: int, lead_steps: int) -> pd.DataFrame:
    required = {
        "agent_context_nowcast_air_temperature_c",
        "agent_context_quality_met_station_core",
        "snow_surface_temperature_c",
        "wind_speed_ms",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    cold_risk = 1.0 - np.clip(
        frame["agent_context_quality_met_station_core"].to_numpy(dtype=float),
        0.0,
        1.0,
    )
    rng = np.random.default_rng(190001 + int(seed))
    residual = np.zeros(len(frame), dtype=float)
    rho = 0.96
    sigma = 0.45
    for index in range(1, len(residual)):
        residual[index] = rho * residual[index - 1] + sigma * rng.normal()
    delayed = np.zeros_like(residual)
    if lead_steps < 0:
        raise ValueError("lead_steps must be non-negative")
    if lead_steps == 0:
        delayed[:] = residual
    elif lead_steps < len(residual):
        delayed[lead_steps:] = residual[:-lead_steps]

    innovation = 0.90 * cold_risk * delayed
    out = frame.copy()
    out["wind_speed_ms"] = np.clip(
        frame["wind_speed_ms"].to_numpy(dtype=float) + innovation,
        0.0,
        45.0,
    )
    out["generator_cold_wind_innovation"] = innovation
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lead-steps", type=int, default=6)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    out = build(frame, seed=args.seed, lead_steps=args.lead_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    metadata = {
        "generator": Path(__file__).name,
        "seed": args.seed,
        "lead_steps": args.lead_steps,
        "rho": 0.96,
        "innovation_sigma_ms": 0.45,
        "cold_risk_scale_ms": 0.90,
        "policy_uses_exact_test_labels": False,
        "policy_uses_generator_innovation_column": False,
        "input_columns_used": [
            "agent_context_quality_met_station_core",
            "wind_speed_ms",
        ],
        "target_column_modified": "wind_speed_ms",
        "audit_column": "generator_cold_wind_innovation",
        "innovation_mean_ms": float(out["generator_cold_wind_innovation"].mean()),
        "innovation_std_ms": float(out["generator_cold_wind_innovation"].std()),
        "innovation_abs_p95_ms": float(
            out["generator_cold_wind_innovation"].abs().quantile(0.95)
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
