"""Screen a documented cumulative-energy envelope before any learning.

This script is a policy-free audit.  It enumerates all 32 optional subsets of
the documented five-channel system, propagates a declared battery state, and
reports whether the resulting resource frontier varies with the meteorology.
It does not fit a forecaster, select a policy, or tune a scheduling budget.

The default generation model is deliberately conservative and explicit:

* PV output is the rated 600 W module scaled by irradiance / 1000 W m-2 and a
  fixed 0.80 system derating.
* Wind output is zero unless ``--include-wind`` is supplied.  The optional
  wind branch uses a fixed cubic cut-in/rated-speed envelope and is reported as
  a sensitivity analysis, not installed-turbine telemetry.
* The fixed load is the documented logger/compute/camera auxiliary envelope
  plus the selected channel loads from ``v2.entity_energy``.

These assumptions are recorded in the output manifest so that a failed screen
closes this route without encouraging post-hoc scenario tuning.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v2.entity_energy import CHANNEL_DEVICE, system_power_w
from v2.entity_resource_envelope import FIXED_AUXILIARY_POWER_W


CHANNELS = tuple(CHANNEL_DEVICE)
LOGGER_LOAD_W = float(system_power_w([])["steady_power_w"])


def _generation(
    truth: pd.DataFrame,
    *,
    include_wind: bool,
    pv_derating: float,
    wind_cut_in_ms: float,
    wind_rated_ms: float,
    wind_cut_out_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    irradiance = np.clip(truth["solar_radiation_wm2"].to_numpy(dtype=float), 0.0, None)
    pv_w = np.minimum(600.0, 600.0 * irradiance / 1000.0 * float(pv_derating))
    wind_w = np.zeros_like(pv_w)
    if include_wind:
        wind = np.clip(truth["wind_speed_ms"].to_numpy(dtype=float), 0.0, None)
        usable = (wind >= float(wind_cut_in_ms)) & (wind < float(wind_cut_out_ms))
        fraction = np.clip(
            (wind - float(wind_cut_in_ms))
            / max(float(wind_rated_ms) - float(wind_cut_in_ms), 1e-9),
            0.0,
            1.0,
        )
        wind_w = np.where(usable, 400.0 * fraction**3, 0.0)
    return pv_w + wind_w, pv_w, wind_w


def _hourly(truth: pd.DataFrame, source_period_s: int) -> pd.DataFrame:
    stride = max(1, int(round(3600.0 / float(source_period_s))))
    sampled = truth.iloc[::stride].reset_index(drop=True).copy()
    if sampled.empty:
        raise ValueError("hourly sampling produced no rows")
    return sampled


def _candidate_rows() -> list[tuple[int, tuple[str, ...], float]]:
    rows = []
    action_idx = 0
    for size in range(len(CHANNELS) + 1):
        for subset in itertools.combinations(CHANNELS, size):
            load = float(FIXED_AUXILIARY_POWER_W + system_power_w(list(subset))["steady_power_w"])
            rows.append((action_idx, tuple(subset), load))
            action_idx += 1
    return rows


def _run(
    truth: pd.DataFrame,
    *,
    capacity_wh: float,
    initial_soc_wh: float,
    reserve_wh: float,
    source_period_s: int,
    include_wind: bool,
    pv_derating: float,
    wind_cut_in_ms: float,
    wind_rated_ms: float,
    wind_cut_out_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    sampled = _hourly(truth, source_period_s)
    generation_w, pv_w, wind_w = _generation(
        sampled,
        include_wind=include_wind,
        pv_derating=pv_derating,
        wind_cut_in_ms=wind_cut_in_ms,
        wind_rated_ms=wind_rated_ms,
        wind_cut_out_ms=wind_cut_out_ms,
    )
    dt_h = float(source_period_s) / 3600.0 * max(1, int(round(3600.0 / float(source_period_s))))
    # The sampled series is hourly by construction; make this explicit rather
    # than silently inheriting an experiment-specific environment setting.
    dt_h = 1.0
    candidates = _candidate_rows()
    fixed_load = float(FIXED_AUXILIARY_POWER_W + LOGGER_LOAD_W)

    # The common baseline trajectory is used only to expose the one-step
    # resource frontier. Candidate-specific trajectories below provide the
    # stricter fixed-schedule support check.
    baseline_soc = np.empty(len(sampled), dtype=float)
    soc = float(np.clip(initial_soc_wh, 0.0, capacity_wh))
    for row_idx, harvest_w in enumerate(generation_w):
        soc = float(np.clip(soc + float(harvest_w - fixed_load) * dt_h, 0.0, capacity_wh))
        baseline_soc[row_idx] = soc

    candidate_rows: list[dict[str, object]] = []
    frontier_rows: list[dict[str, object]] = []
    frontier_counts = np.zeros(len(sampled), dtype=int)
    for action_idx, subset, load in candidates:
        soc = float(np.clip(initial_soc_wh, 0.0, capacity_wh))
        soc_trace = np.empty(len(sampled), dtype=float)
        reserve_violations = np.zeros(len(sampled), dtype=bool)
        for row_idx, harvest_w in enumerate(generation_w):
            soc_unclipped = soc + float(harvest_w - load) * dt_h
            reserve_violations[row_idx] = soc_unclipped < float(reserve_wh) - 1e-9
            soc = float(np.clip(soc_unclipped, 0.0, capacity_wh))
            soc_trace[row_idx] = soc
            if baseline_soc[row_idx] + float(harvest_w - load) * dt_h >= float(reserve_wh) - 1e-9:
                frontier_counts[row_idx] += 1
        candidate_rows.append(
            {
                "action_idx": int(action_idx),
                "selected_channels": ";".join(subset),
                "selected_count": int(len(subset)),
                "load_w": float(load),
                "fixed_schedule_feasible_fraction": float(np.mean(~reserve_violations)),
                "fixed_schedule_final_soc_wh": float(soc_trace[-1]),
                "fixed_schedule_min_soc_wh": float(np.min(soc_trace)),
                "fixed_schedule_reserve_violations": int(np.sum(reserve_violations)),
            }
        )

    generation_regime = np.where(generation_w <= 1e-9, "zero", np.where(generation_w < 50.0, "low", "high"))
    soc_regime = np.where(
        baseline_soc < 0.25 * capacity_wh,
        "low",
        np.where(baseline_soc < 0.75 * capacity_wh, "mid", "high"),
    )
    for row_idx in range(len(sampled)):
        frontier_rows.append(
            {
                "row_idx": int(row_idx),
                "timestamp": sampled.iloc[row_idx].get("timestamp", row_idx),
                "generation_w": float(generation_w[row_idx]),
                "pv_w": float(pv_w[row_idx]),
                "wind_w": float(wind_w[row_idx]),
                "baseline_soc_wh": float(baseline_soc[row_idx]),
                "generation_regime": str(generation_regime[row_idx]),
                "soc_regime": str(soc_regime[row_idx]),
                "feasible_action_count": int(frontier_counts[row_idx]),
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows)
    frontier_frame = pd.DataFrame(frontier_rows)
    grouped = (
        frontier_frame.groupby(["generation_regime", "soc_regime"], dropna=False)["feasible_action_count"]
        .agg(["count", "min", "max", "mean"])
        .reset_index()
    )
    summary = {
        "rows": int(len(sampled)),
        "candidate_count": int(len(candidates)),
        "capacity_wh": float(capacity_wh),
        "initial_soc_wh": float(initial_soc_wh),
        "reserve_wh": float(reserve_wh),
        "fixed_load_w": float(fixed_load),
        "mean_generation_w": float(np.mean(generation_w)),
        "generation_zero_fraction": float(np.mean(generation_w <= 1e-9)),
        "baseline_soc_min_wh": float(np.min(baseline_soc)),
        "baseline_soc_max_wh": float(np.max(baseline_soc)),
        "frontier_count_min": int(np.min(frontier_counts)),
        "frontier_count_max": int(np.max(frontier_counts)),
        "frontier_count_unique": int(np.unique(frontier_counts).size),
        "frontier_time_varying_fraction": float(np.mean(frontier_counts != frontier_counts[0])),
        "full_horizon_fixed_candidates": int(np.sum(candidate_frame["fixed_schedule_reserve_violations"] == 0)),
        "grouped_frontier": grouped.to_dict(orient="records"),
    }
    return candidate_frame, frontier_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-period-s", type=int, default=3600)
    parser.add_argument("--capacity-wh", type=float, default=8640.0)
    parser.add_argument("--initial-soc-wh", type=float, default=8640.0)
    parser.add_argument("--reserve-wh", type=float, default=0.0)
    parser.add_argument("--pv-derating", type=float, default=0.80)
    parser.add_argument("--include-wind", action="store_true")
    parser.add_argument("--wind-cut-in-ms", type=float, default=3.0)
    parser.add_argument("--wind-rated-ms", type=float, default=12.0)
    parser.add_argument("--wind-cut-out-ms", type=float, default=25.0)
    args = parser.parse_args()

    truth = pd.read_csv(args.input_truth)
    required = {"solar_radiation_wm2", "wind_speed_ms"}
    missing = sorted(required - set(truth.columns))
    if missing:
        raise ValueError(f"truth is missing required generation columns: {missing}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_frame, frontier_frame, summary = _run(
        truth,
        capacity_wh=float(args.capacity_wh),
        initial_soc_wh=float(args.initial_soc_wh),
        reserve_wh=float(args.reserve_wh),
        source_period_s=int(args.source_period_s),
        include_wind=bool(args.include_wind),
        pv_derating=float(args.pv_derating),
        wind_cut_in_ms=float(args.wind_cut_in_ms),
        wind_rated_ms=float(args.wind_rated_ms),
        wind_cut_out_ms=float(args.wind_cut_out_ms),
    )
    candidate_frame.to_csv(output_dir / "candidate_energy_support.csv", index=False)
    frontier_frame.to_csv(output_dir / "energy_frontier_trace.csv", index=False)
    manifest = {
        "script": "180_screen_documented_energy_frontier.py",
        "input_truth": str(args.input_truth),
        "source_period_s": int(args.source_period_s),
        "battery": {"capacity_wh": float(args.capacity_wh), "initial_soc_wh": float(args.initial_soc_wh), "reserve_wh": float(args.reserve_wh)},
        "pv": {"rated_w": 600.0, "reference_irradiance_wm2": 1000.0, "derating": float(args.pv_derating)},
        "wind": {"included": bool(args.include_wind), "rated_w": 400.0, "cut_in_ms": float(args.wind_cut_in_ms), "rated_speed_ms": float(args.wind_rated_ms), "cut_out_ms": float(args.wind_cut_out_ms), "status": "sensitivity_model_not_installed_telemetry"},
        "loads": {"fixed_auxiliary_power_w": float(FIXED_AUXILIARY_POWER_W), "logger_power_w": float(LOGGER_LOAD_W), "channel_power_source": "src/v2/entity_energy.py"},
        "interpretation": "truth/resource-only screen; no forecaster, policy, or post-hoc scene selection",
        "summary": summary,
    }
    (output_dir / "energy_frontier_summary.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
