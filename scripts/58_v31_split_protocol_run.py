#!/usr/bin/env python
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def partition_bounds(steps: int, ratios: tuple[float, float, float, float]) -> dict[str, tuple[int, int]]:
    total = float(sum(ratios))
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to one, got {total}")
    edges = [0]
    cumulative = 0.0
    for ratio in ratios[:-1]:
        cumulative += float(ratio)
        edges.append(int(round(int(steps) * cumulative)))
    edges.append(int(steps))
    names = ("oracle_pretrain", "rl_train", "validation", "final_test")
    bounds = {name: (int(edges[idx]), int(edges[idx + 1])) for idx, name in enumerate(names)}
    if any(end <= start for start, end in bounds.values()):
        raise ValueError(f"All partitions must be nonempty, got {bounds}")
    return bounds


def non_overlapping_starts(
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    start, end = (int(bounds[0]), int(bounds[1]))
    required_span = int(count) * int(window_steps) + int(horizon) + 1
    available_span = end - start
    if available_span < required_span:
        raise ValueError(
            f"Partition [{start}, {end}) cannot contain {count} non-overlapping "
            f"{window_steps}-step windows plus horizon {horizon}"
        )
    rng = np.random.default_rng(int(seed))
    slack = int(available_span - required_span)
    gaps = rng.multinomial(slack, np.full(int(count) + 1, 1.0 / float(int(count) + 1)))
    starts: list[int] = []
    cursor = start + int(gaps[0])
    for idx in range(int(count)):
        starts.append(int(cursor))
        cursor += int(window_steps) + int(gaps[idx + 1])
    return tuple(starts)


def ensure_truth(args: argparse.Namespace, truth_path: Path) -> Path:
    if truth_path.exists():
        return truth_path
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "20_build_public_weather_truth.py"),
        "--antaws-root",
        str(args.antaws_root),
        "--stations",
        *[str(station) for station in args.stations],
        "--steps",
        str(int(args.truth_steps)),
        "--freq-s",
        str(int(args.freq_s)),
        "--seed",
        str(int(args.seed)),
        "--blowing-snow-event-coverage",
        str(float(args.event_coverage)),
        "--blowing-snow-event-model",
        "semi_markov",
        "--blowing-snow-min-duration-steps",
        str(int(args.min_duration)),
        "--blowing-snow-max-duration-steps",
        str(int(args.max_duration)),
        "--blowing-snow-min-gap-steps",
        str(int(args.min_gap)),
        "--blowing-snow-lead-steps",
        str(int(args.lead_steps)),
        "--blowing-snow-wind-margin-ms",
        str(float(args.wind_margin_ms)),
        "--cred-hysteresis-on",
        str(float(args.cred_hysteresis_on)),
        "--cred-hysteresis-off",
        str(float(args.cred_hysteresis_off)),
        "--flux-wind-exponent",
        str(float(args.flux_wind_exponent)),
        "--event-microstructure-sigma",
        str(float(args.event_microstructure_sigma)),
        "--event-microstructure-alpha",
        str(float(args.event_microstructure_alpha)),
        "--event-microstructure-diameter-scale",
        str(float(args.event_microstructure_diameter_scale)),
        "--event-microstructure-velocity-scale",
        str(float(args.event_microstructure_velocity_scale)),
        "--event-particle-microstructure-correlation",
        str(float(args.event_particle_microstructure_correlation)),
        "--event-subtype-assignment",
        str(args.event_subtype_assignment),
        "--event-subtype-particle-min-parsivel-availability",
        str(float(args.event_subtype_particle_min_parsivel_availability)),
        "--event-subtype-particle-prob",
        str(float(args.event_subtype_particle_prob)),
        "--event-subtype-flux-prob",
        str(float(args.event_subtype_flux_prob)),
        "--event-subtype-thermal-prob",
        str(float(args.event_subtype_thermal_prob)),
        "--event-subtype-particle-flux-multiplier",
        str(float(args.event_subtype_particle_flux_multiplier)),
        "--event-subtype-flux-multiplier",
        str(float(args.event_subtype_flux_multiplier)),
        "--event-subtype-thermal-flux-multiplier",
        str(float(args.event_subtype_thermal_flux_multiplier)),
        "--event-subtype-particle-diameter-shift-mm",
        str(float(args.event_subtype_particle_diameter_shift_mm)),
        "--event-subtype-particle-velocity-boost-ms",
        str(float(args.event_subtype_particle_velocity_boost_ms)),
        "--event-subtype-flux-diameter-shift-mm",
        str(float(args.event_subtype_flux_diameter_shift_mm)),
        "--event-subtype-flux-velocity-boost-ms",
        str(float(args.event_subtype_flux_velocity_boost_ms)),
        "--event-subtype-thermal-surface-drop-c",
        str(float(args.event_subtype_thermal_surface_drop_c)),
        "--event-subtype-particle-humidity-boost-pct",
        str(float(args.event_subtype_particle_humidity_boost_pct)),
        "--event-subtype-flux-wind-boost-ms",
        str(float(args.event_subtype_flux_wind_boost_ms)),
        "--event-subtype-thermal-air-temp-drop-c",
        str(float(args.event_subtype_thermal_air_temp_drop_c)),
        "--event-subtype-latent-alpha",
        str(float(args.event_subtype_latent_alpha)),
        "--event-subtype-particle-latent-diameter-scale-mm",
        str(float(args.event_subtype_particle_latent_diameter_scale_mm)),
        "--event-subtype-particle-latent-velocity-scale-ms",
        str(float(args.event_subtype_particle_latent_velocity_scale_ms)),
        "--event-subtype-flux-latent-sigma",
        str(float(args.event_subtype_flux_latent_sigma)),
        "--event-subtype-flux-latent-linear-scale",
        str(float(args.event_subtype_flux_latent_linear_scale)),
        "--event-subtype-flux-latent-linear-offset",
        str(float(args.event_subtype_flux_latent_linear_offset)),
        "--event-subtype-flux-latent-linear-clip",
        str(float(args.event_subtype_flux_latent_linear_clip)),
        "--event-subtype-thermal-latent-surface-scale-c",
        str(float(args.event_subtype_thermal_latent_surface_scale_c)),
        "--event-subtype-latent-target-lag-steps",
        str(int(args.event_subtype_latent_target_lag_steps)),
        "--event-subtype-context-lead-steps",
        str(int(args.event_subtype_context_lead_steps)),
        "--event-subtype-context-noise-std",
        str(float(args.event_subtype_context_noise_std)),
        "--event-subtype-context-latent-strength",
        str(float(args.event_subtype_context_latent_strength)),
        "--out",
        str(truth_path),
        "--report-dir",
        str(Path(args.out_dir) / "dataset_validation"),
    ]
    if bool(args.event_subtypes_enabled):
        cmd.append("--event-subtypes-enabled")
    subprocess.run(cmd, check=True)
    return truth_path


def rich_non_overlapping_starts(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    count: int,
    selection: str,
    stride: int,
    seed: int,
) -> tuple[int, ...]:
    start_min, end = (int(bounds[0]), int(bounds[1]))
    required_span = int(count) * int(window_steps) + int(horizon) + 1
    available_span = int(end) - int(start_min)
    if available_span < required_span:
        raise ValueError(
            f"Partition [{start_min}, {end}) cannot contain {count} non-overlapping "
            f"{window_steps}-step windows plus horizon {horizon}"
        )
    start_max = int(end) - int(window_steps) - int(horizon) - 1
    if start_max < start_min:
        raise ValueError(
            f"Partition [{start_min}, {end}) cannot contain {count} "
            f"{window_steps}-step windows plus horizon {horizon}"
        )
    event_flags = (
        truth["event_flag"].astype(bool).to_numpy()
        if "event_flag" in truth.columns
        else np.zeros(len(truth), dtype=bool)
    )
    flux = (
        np.asarray(truth["snow_mass_flux_kg_m2_s"], dtype=float)
        if "snow_mass_flux_kg_m2_s" in truth.columns
        else np.zeros(len(truth), dtype=float)
    )
    flux = np.maximum(0.0, np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0))
    starts = np.unique(
        np.concatenate(
            [
                np.arange(start_min, start_max + 1, max(1, int(stride)), dtype=int),
                np.asarray([start_min, start_max], dtype=int),
            ]
        )
    )
    if len(starts) == 0:
        return (int(start_min),)
    rng = np.random.default_rng(int(seed))
    flux_scale = max(float(np.percentile(flux, 95)) if flux.size else 0.0, 1e-12)
    rows: list[tuple[int, float]] = []
    for start in starts:
        window_end = int(start) + int(window_steps)
        event_rate = float(np.mean(event_flags[int(start) : window_end])) if window_end > start else 0.0
        if str(selection) == "event_rich":
            score = event_rate
        elif str(selection) == "event_transport_rich":
            window_flux = float(np.mean(flux[int(start) : window_end])) if window_end > start else 0.0
            score = event_rate * (1.0 + float(np.clip(window_flux / flux_scale, 0.0, 4.0)))
        else:
            raise ValueError(f"Unknown eval start selection: {selection}")
        rows.append((int(start), float(score + 1e-9 * rng.random())))
    candidates = sorted(rows, key=lambda item: item[0])
    n_candidates = len(candidates)
    previous: list[int] = []
    for idx, (start, _) in enumerate(candidates):
        prev_idx = idx - 1
        while prev_idx >= 0 and int(start) - int(candidates[prev_idx][0]) < int(window_steps):
            prev_idx -= 1
        previous.append(prev_idx)
    neg_inf = -1e300
    dp = [[neg_inf] * (int(count) + 1) for _ in range(n_candidates + 1)]
    keep = [[False] * (int(count) + 1) for _ in range(n_candidates + 1)]
    for idx in range(n_candidates + 1):
        dp[idx][0] = 0.0
    for idx in range(1, n_candidates + 1):
        start, score = candidates[idx - 1]
        prev_row = previous[idx - 1] + 1
        for chosen in range(1, int(count) + 1):
            skip_score = dp[idx - 1][chosen]
            take_base = dp[prev_row][chosen - 1]
            take_score = take_base + float(score) if take_base > neg_inf / 2 else neg_inf
            if take_score > skip_score:
                dp[idx][chosen] = take_score
                keep[idx][chosen] = True
            else:
                dp[idx][chosen] = skip_score
    if dp[n_candidates][int(count)] <= neg_inf / 2:
        raise ValueError(
            f"Could not select {count} non-overlapping {selection} starts in "
            f"partition [{start_min}, {end}) with stride {stride}"
        )
    selected: list[int] = []
    idx = n_candidates
    chosen = int(count)
    while chosen > 0 and idx > 0:
        if keep[idx][chosen]:
            start, _ = candidates[idx - 1]
            selected.append(int(start))
            idx = previous[idx - 1] + 1
            chosen -= 1
        else:
            idx -= 1
    if len(selected) != int(count):
        raise RuntimeError(f"Backtracking selected {len(selected)} starts, expected {count}")
    return tuple(sorted(int(value) for value in selected))


def subtype_balanced_non_overlapping_starts(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    count: int,
    selection: str,
    stride: int,
    seed: int,
) -> tuple[int, ...]:
    if "event_subtype_id" not in truth.columns:
        return rich_non_overlapping_starts(
            truth,
            bounds=bounds,
            window_steps=int(window_steps),
            horizon=int(horizon),
            count=int(count),
            selection="event_transport_rich" if "transport" in str(selection) else "event_rich",
            stride=int(stride),
            seed=int(seed),
        )

    start_min, end = (int(bounds[0]), int(bounds[1]))
    required_span = int(count) * int(window_steps) + int(horizon) + 1
    available_span = int(end) - int(start_min)
    if available_span < required_span:
        raise ValueError(
            f"Partition [{start_min}, {end}) cannot contain {count} non-overlapping "
            f"{window_steps}-step windows plus horizon {horizon}"
        )
    start_max = int(end) - int(window_steps) - int(horizon) - 1
    if start_max < start_min:
        raise ValueError(
            f"Partition [{start_min}, {end}) cannot contain {count} "
            f"{window_steps}-step windows plus horizon {horizon}"
        )
    subtype_ids = truth["event_subtype_id"].astype(int).to_numpy()
    event_flags = (
        truth["event_flag"].astype(bool).to_numpy()
        if "event_flag" in truth.columns
        else subtype_ids > 0
    )
    flux = (
        np.asarray(truth["snow_mass_flux_kg_m2_s"], dtype=float)
        if "snow_mass_flux_kg_m2_s" in truth.columns
        else np.zeros(len(truth), dtype=float)
    )
    flux = np.maximum(0.0, np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0))
    starts = np.unique(
        np.concatenate(
            [
                np.arange(start_min, start_max + 1, max(1, int(stride)), dtype=int),
                np.asarray([start_min, start_max], dtype=int),
            ]
        )
    )
    if len(starts) == 0:
        return (int(start_min),)
    rng = np.random.default_rng(int(seed))
    flux_scale = max(float(np.percentile(flux, 95)) if flux.size else 0.0, 1e-12)
    rows: list[tuple[int, float, int]] = []
    for start in starts:
        window_end = int(start) + int(window_steps)
        window_subtypes = subtype_ids[int(start) : window_end]
        subtype_counts = np.asarray(
            [np.sum(window_subtypes == subtype_id) for subtype_id in (1, 2, 3)],
            dtype=float,
        )
        dominant_idx = int(np.argmax(subtype_counts))
        dominant_subtype = int(dominant_idx + 1) if float(subtype_counts[dominant_idx]) > 0.0 else 0
        event_rate = float(np.mean(event_flags[int(start) : window_end])) if window_end > start else 0.0
        subtype_total = float(np.sum(subtype_counts))
        subtype_fraction = subtype_counts / max(subtype_total, 1.0)
        entropy = 0.0
        positive = subtype_fraction[subtype_fraction > 0]
        if positive.size:
            entropy = float(-np.sum(positive * np.log(positive)) / np.log(3.0))
        window_flux = float(np.mean(flux[int(start) : window_end])) if window_end > start else 0.0
        transport_score = event_rate * (1.0 + float(np.clip(window_flux / flux_scale, 0.0, 4.0)))
        base_score = transport_score if "transport" in str(selection) else event_rate
        score = (
            float(base_score)
            + 0.35 * float(entropy)
            + 0.20 * float(np.count_nonzero(subtype_counts > 0.05 * float(window_steps)))
            + 0.10 * float(subtype_counts[dominant_idx] / max(float(window_steps), 1.0))
            + 1e-9 * float(rng.random())
        )
        rows.append((int(start), float(score), int(dominant_subtype)))
    candidates = sorted(rows, key=lambda item: item[0])
    n_candidates = len(candidates)
    previous: list[int] = []
    for idx, (start, _, _) in enumerate(candidates):
        prev_idx = idx - 1
        while prev_idx >= 0 and int(start) - int(candidates[prev_idx][0]) < int(window_steps):
            prev_idx -= 1
        previous.append(prev_idx)

    neg_inf = -1e300

    def solve(min_per_subtype: int) -> tuple[float, tuple[int, ...]]:
        @functools.lru_cache(maxsize=None)
        def best(
            idx: int,
            chosen: int,
            c1: int,
            c2: int,
            c3: int,
        ) -> tuple[float, tuple[int, ...]]:
            if chosen == int(count):
                if c1 >= min_per_subtype and c2 >= min_per_subtype and c3 >= min_per_subtype:
                    return 0.0, tuple()
                return neg_inf, tuple()
            if idx <= 0:
                return neg_inf, tuple()
            remaining = int(count) - int(chosen)
            if idx < remaining:
                return neg_inf, tuple()
            skip_score, skip_values = best(idx - 1, chosen, c1, c2, c3)
            start, score, dominant_subtype = candidates[idx - 1]
            nc1 = min(int(count), c1 + (1 if dominant_subtype == 1 else 0))
            nc2 = min(int(count), c2 + (1 if dominant_subtype == 2 else 0))
            nc3 = min(int(count), c3 + (1 if dominant_subtype == 3 else 0))
            take_base, take_values = best(previous[idx - 1] + 1, chosen + 1, nc1, nc2, nc3)
            take_score = float(score) + take_base if take_base > neg_inf / 2 else neg_inf
            if take_score > skip_score:
                return take_score, (int(start), *take_values)
            return skip_score, skip_values

        return best(n_candidates, 0, 0, 0, 0)

    desired_min = max(1, int(count) // 4)
    for min_per_subtype in range(desired_min, -1, -1):
        score, selected = solve(int(min_per_subtype))
        if score > neg_inf / 2 and len(selected) == int(count):
            return tuple(sorted(int(value) for value in selected))
    raise ValueError(
        f"Could not select {count} non-overlapping subtype-balanced starts in "
        f"partition [{start_min}, {end}) with stride {stride}"
    )


def event_fraction_starts(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    count: int,
    event_fraction: float,
    stride: int,
    seed: int,
) -> tuple[int, ...]:
    start_min, end = (int(bounds[0]), int(bounds[1]))
    start_max = int(end) - int(window_steps) - int(horizon) - 1
    if start_max < start_min:
        raise ValueError(
            f"Partition [{start_min}, {end}) cannot contain {count} "
            f"{window_steps}-step windows plus horizon {horizon}"
        )
    event_flags = (
        truth["event_flag"].astype(bool).to_numpy()
        if "event_flag" in truth.columns
        else np.zeros(len(truth), dtype=bool)
    )
    starts = np.unique(
        np.concatenate(
            [
                np.arange(start_min, start_max + 1, max(1, int(stride)), dtype=int),
                np.asarray([start_min, start_max], dtype=int),
            ]
        )
    )
    rng = np.random.default_rng(int(seed))
    scored: list[tuple[int, float]] = []
    for start in starts:
        window_end = int(start) + int(window_steps)
        event_rate = float(np.mean(event_flags[int(start) : window_end])) if window_end > start else 0.0
        scored.append((int(start), float(event_rate + 1e-9 * rng.random())))
    n_event = int(round(float(np.clip(event_fraction, 0.0, 1.0)) * int(count)))
    n_event = max(0, min(int(count), n_event))

    def compatible(start: int, fixed: tuple[int, ...]) -> bool:
        return all(abs(int(start) - int(other)) >= int(window_steps) for other in fixed)

    def ranked_selections(
        candidates: list[tuple[int, float]],
        needed: int,
        fixed: tuple[int, ...],
        *,
        limit: int = 4096,
    ) -> list[tuple[int, ...]]:
        out: list[tuple[int, ...]] = []

        def rec(pos: int, chosen: tuple[int, ...]) -> None:
            if len(out) >= int(limit):
                return
            if len(chosen) == int(needed):
                out.append(chosen)
                return
            remaining = int(needed) - len(chosen)
            if len(candidates) - int(pos) < remaining:
                return
            for idx in range(int(pos), len(candidates)):
                start = int(candidates[idx][0])
                if compatible(start, fixed + chosen):
                    rec(idx + 1, chosen + (start,))

        rec(0, tuple())
        return out

    high_ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    low_ranked = sorted(scored, key=lambda item: item[1])
    selected: tuple[int, ...] | None = None
    event_targets = [n_event]
    for offset in range(1, int(count) + 1):
        event_targets.extend([n_event - offset, n_event + offset])
    for event_needed in event_targets:
        if event_needed < 0 or event_needed > int(count):
            continue
        calm_needed = int(count) - int(event_needed)
        for event_choice in ranked_selections(high_ranked, int(event_needed), tuple()):
            calm_choices = ranked_selections(low_ranked, calm_needed, tuple(event_choice), limit=1)
            if calm_choices:
                selected = tuple(sorted(tuple(event_choice) + tuple(calm_choices[0])))
                break
        if selected is not None:
            break
    if selected is None or len(selected) != int(count):
        raise ValueError(
            f"Could not select {count} non-overlapping event-fraction starts in "
            f"partition [{start_min}, {end}) with stride {stride}"
        )
    return tuple(int(value) for value in selected)


def summarize_eval_windows(
    truth: pd.DataFrame,
    *,
    starts: tuple[int, ...],
    window_steps: int,
) -> dict[str, float | int]:
    if not starts:
        return {"count": 0, "event_rate_mean": float("nan"), "flux_mean": float("nan")}
    event_flags = (
        truth["event_flag"].astype(bool).to_numpy()
        if "event_flag" in truth.columns
        else np.zeros(len(truth), dtype=bool)
    )
    flux = (
        np.asarray(truth["snow_mass_flux_kg_m2_s"], dtype=float)
        if "snow_mass_flux_kg_m2_s" in truth.columns
        else np.zeros(len(truth), dtype=float)
    )
    subtype_ids = (
        truth["event_subtype_id"].astype(int).to_numpy()
        if "event_subtype_id" in truth.columns
        else np.zeros(len(truth), dtype=int)
    )
    rates = []
    flux_values = []
    subtype_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for start in starts:
        end = int(start) + int(window_steps)
        rates.append(float(np.mean(event_flags[int(start) : end])) if end > start else 0.0)
        flux_values.append(float(np.mean(flux[int(start) : end])) if end > start else 0.0)
        values, counts = np.unique(subtype_ids[int(start) : end], return_counts=True)
        for value, count in zip(values, counts):
            subtype_counts[int(value)] = subtype_counts.get(int(value), 0) + int(count)
    total_steps = max(1, int(window_steps) * int(len(starts)))
    return {
        "count": int(len(starts)),
        "event_rate_mean": float(np.mean(rates)),
        "event_rate_min": float(np.min(rates)),
        "event_rate_max": float(np.max(rates)),
        "flux_mean": float(np.mean(flux_values)),
        "subtype_calm_fraction": float(subtype_counts.get(0, 0) / total_steps),
        "subtype_particle_fraction": float(subtype_counts.get(1, 0) / total_steps),
        "subtype_flux_fraction": float(subtype_counts.get(2, 0) / total_steps),
        "subtype_thermal_fraction": float(subtype_counts.get(3, 0) / total_steps),
    }


def append_option(cmd: list[str], flag: str, values: list[str] | tuple[str, ...] | None) -> None:
    if values:
        cmd.extend([flag, *values])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one V3.1 experiment with explicit chronological splits.")
    parser.add_argument("--out-dir", default="reports/v31_split_protocol/budget1p70_seed41")
    parser.add_argument("--truth-csv", default=None)
    parser.add_argument(
        "--control-source-run-dir",
        default=None,
        help="Reuse and validate truth, frozen forecaster, masks, and validation assets from this run.",
    )
    parser.add_argument("--validate-control-source-only", action="store_true")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced.yaml")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--policy-seed", type=int, default=None)
    parser.add_argument("--budget", type=float, default=1.70)
    parser.add_argument("--startup-peak-budget", type=float, default=3.20)
    parser.add_argument("--truth-steps", type=int, default=90000)
    parser.add_argument("--freq-s", type=int, default=3600)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--split-ratios", nargs=4, type=float, default=[0.35, 0.50, 0.075, 0.075])
    parser.add_argument("--event-coverage", type=float, default=0.28)
    parser.add_argument("--min-duration", type=int, default=12)
    parser.add_argument("--max-duration", type=int, default=24)
    parser.add_argument("--min-gap", type=int, default=4)
    parser.add_argument("--lead-steps", type=int, default=6)
    parser.add_argument("--wind-margin-ms", type=float, default=1.2)
    parser.add_argument("--cred-hysteresis-on", type=float, default=0.6)
    parser.add_argument("--cred-hysteresis-off", type=float, default=0.3)
    parser.add_argument("--flux-wind-exponent", type=float, default=3.0)
    parser.add_argument("--event-microstructure-sigma", type=float, default=0.0)
    parser.add_argument("--event-microstructure-alpha", type=float, default=0.18)
    parser.add_argument("--event-microstructure-diameter-scale", type=float, default=0.0)
    parser.add_argument("--event-microstructure-velocity-scale", type=float, default=0.0)
    parser.add_argument("--event-particle-microstructure-correlation", type=float, default=1.0)
    parser.add_argument("--event-subtypes-enabled", action="store_true")
    parser.add_argument("--event-subtype-assignment", choices=["random", "stratified"], default="random")
    parser.add_argument("--event-subtype-particle-min-parsivel-availability", type=float, default=0.0)
    parser.add_argument("--event-subtype-particle-prob", type=float, default=0.34)
    parser.add_argument("--event-subtype-flux-prob", type=float, default=0.33)
    parser.add_argument("--event-subtype-thermal-prob", type=float, default=0.33)
    parser.add_argument("--event-subtype-particle-flux-multiplier", type=float, default=0.72)
    parser.add_argument("--event-subtype-flux-multiplier", type=float, default=2.4)
    parser.add_argument("--event-subtype-thermal-flux-multiplier", type=float, default=0.55)
    parser.add_argument("--event-subtype-particle-diameter-shift-mm", type=float, default=0.10)
    parser.add_argument("--event-subtype-particle-velocity-boost-ms", type=float, default=1.3)
    parser.add_argument("--event-subtype-flux-diameter-shift-mm", type=float, default=-0.04)
    parser.add_argument("--event-subtype-flux-velocity-boost-ms", type=float, default=0.7)
    parser.add_argument("--event-subtype-thermal-surface-drop-c", type=float, default=2.0)
    parser.add_argument("--event-subtype-particle-humidity-boost-pct", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-wind-boost-ms", type=float, default=0.0)
    parser.add_argument("--event-subtype-thermal-air-temp-drop-c", type=float, default=0.0)
    parser.add_argument("--event-subtype-latent-alpha", type=float, default=0.18)
    parser.add_argument("--event-subtype-particle-latent-diameter-scale-mm", type=float, default=0.0)
    parser.add_argument("--event-subtype-particle-latent-velocity-scale-ms", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-latent-sigma", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-latent-linear-scale", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-latent-linear-offset", type=float, default=1.5)
    parser.add_argument("--event-subtype-flux-latent-linear-clip", type=float, default=4.0)
    parser.add_argument("--event-subtype-thermal-latent-surface-scale-c", type=float, default=0.0)
    parser.add_argument("--event-subtype-latent-target-lag-steps", type=int, default=0)
    parser.add_argument("--event-subtype-context-lead-steps", type=int, default=0)
    parser.add_argument("--event-subtype-context-noise-std", type=float, default=0.08)
    parser.add_argument("--event-subtype-context-latent-strength", type=float, default=0.0)
    parser.add_argument("--oracle-rollout-steps", type=int, default=7200)
    parser.add_argument("--oracle-type", choices=["linear", "tcn"], default="tcn")
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=6)
    parser.add_argument("--oracle-full-open-repeat", type=int, default=3)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--oracle-loss-clip", type=float, default=10.0)
    parser.add_argument("--oracle-candidate-mask-repeat", type=int, default=0)
    parser.add_argument("--oracle-candidate-mask-limit", type=int, default=0)
    parser.add_argument("--oracle-subtype-teacher-repeat", type=int, default=0)
    parser.add_argument("--oracle-subtype-teacher-lookahead-steps", type=int, default=0)
    parser.add_argument("--oracle-subtype-teacher-calm-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-subtype-teacher-particle-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-subtype-teacher-flux-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-subtype-teacher-thermal-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-device", default="auto")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--policy-checkpoint-source", default=None)
    parser.add_argument(
        "--evaluation-policy-mode",
        choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    parser.add_argument("--evaluation-sampling-seed", type=int, default=None)
    parser.add_argument("--evaluation-sampling-temperature", type=float, default=1.0)
    parser.add_argument("--evaluation-temperature-candidates", nargs="*", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--channel-marginal-entropy-coef", type=float, default=0.0)
    parser.add_argument(
        "--separate-actor-critic-grad-clip",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--awbc-coef", type=float, default=0.1)
    parser.add_argument("--awbc-decay-timesteps", type=int, default=0)
    parser.add_argument("--awbc-label-stride", type=int, default=4)
    parser.add_argument("--awbc-event-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-selection-interval-updates", type=int, default=0)
    parser.add_argument(
        "--checkpoint-selection-score",
        choices=[
            "oracle_loss_mean",
            "oracle_loss_macro_subtype_event",
            "oracle_loss_macro_subtype_event_staticnorm",
            "max_static_ratio",
        ],
        default="oracle_loss_mean",
    )
    parser.add_argument("--bc-pretrain-steps", type=int, default=0)
    parser.add_argument("--bc-pretrain-epochs", type=int, default=4)
    parser.add_argument("--bc-pretrain-batch-size", type=int, default=128)
    parser.add_argument("--bc-pretrain-loss-coef", type=float, default=1.0)
    parser.add_argument(
        "--bc-pretrain-target-mode",
        choices=["hard", "soft_forecast_value", "forecast_value_regression"],
        default="hard",
    )
    parser.add_argument("--bc-soft-temperature", type=float, default=1.0)
    parser.add_argument("--subtype-aux-coef", type=float, default=0.0)
    parser.add_argument("--subtype-aux-classes", type=int, default=4)
    parser.add_argument("--subtype-aux-lookahead-steps", type=int, default=0)
    parser.add_argument("--subtype-action-ce-coef", type=float, default=0.0)
    parser.add_argument(
        "--subtype-action-supervision-mode",
        choices=["exact_action", "positive_sensor_inclusion"],
        default="exact_action",
    )
    parser.add_argument("--subtype-action-event-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-action-margin-coef", type=float, default=0.0)
    parser.add_argument("--subtype-action-margin", type=float, default=0.5)
    parser.add_argument("--subtype-router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-router-min-confidence", type=float, default=0.0)
    parser.add_argument("--subtype-router-low-confidence-action", type=int, default=-1)
    parser.add_argument(
        "--awbc-teacher-mode",
        choices=[
            "oracle_greedy",
            "event_pair",
            "event_cyclic",
            "subtype_auto",
            "subtype_static_auto",
            "context_alert",
            "energy_mpc",
        ],
        default="oracle_greedy",
    )
    parser.add_argument("--awbc-teacher-calm-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-event-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-event-lookahead-steps", type=int, default=0)
    parser.add_argument("--awbc-teacher-alert-threshold", type=float, default=0.5)
    parser.add_argument("--awbc-teacher-energy-mpc-horizon", type=int, default=4)
    parser.add_argument("--awbc-teacher-energy-mpc-soc-bins", type=int, default=16)
    parser.add_argument("--awbc-teacher-energy-mpc-low-soc-ratio", type=float, default=0.25)
    parser.add_argument("--awbc-teacher-energy-mpc-high-soc-ratio", type=float, default=0.75)
    parser.add_argument("--awbc-teacher-energy-mpc-terminal-soc-weight", type=float, default=0.0)
    parser.add_argument("--awbc-teacher-energy-mpc-max-actions", type=int, default=0)
    parser.add_argument("--awbc-teacher-energy-mpc-low-power-action", type=int, default=-1)
    parser.add_argument("--awbc-teacher-calm-pool-spec", default=None)
    parser.add_argument("--awbc-teacher-event-pool-spec", default=None)
    parser.add_argument("--awbc-teacher-subtype-calm-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-subtype-particle-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-subtype-flux-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-subtype-thermal-sensors", nargs="*", default=None)
    parser.add_argument(
        "--awbc-teacher-auto-score-mode",
        choices=["raw", "staticnorm"],
        default="raw",
    )
    parser.add_argument("--awbc-teacher-dwell-steps", type=int, default=1)
    parser.add_argument("--prior-kl-coef", type=float, default=1.0)
    parser.add_argument("--greedy-lookahead-steps", type=int, default=4)
    parser.add_argument("--event-start-prob", type=float, default=0.67)
    parser.add_argument("--event-aware-critic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trainable-action-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nonlinear-action-embedding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-gated-actor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--context-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--context-feature-dim", type=int, default=0)
    parser.add_argument("--context-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--context-fusion-mode",
        choices=["concat", "gated_add", "subtype_moe"],
        default="concat",
    )
    parser.add_argument("--context-layer-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--soc-aux-horizon", type=int, default=0)
    parser.add_argument("--soc-aux-coef", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-episode-len", type=int, default=512)
    parser.add_argument("--use-candidate-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-prior-scale", type=float, default=2.0)
    parser.add_argument("--candidate-prior-steps", type=int, default=512)
    parser.add_argument("--candidate-prior-rollouts", type=int, default=4)
    parser.add_argument("--static-selection-steps", type=int, default=512)
    parser.add_argument(
        "--static-selection-score",
        choices=[
            "oracle_loss_mean",
            "oracle_loss_macro_subtype_event",
            "oracle_loss_macro_subtype_event_staticnorm",
        ],
        default="oracle_loss_mean",
    )
    parser.add_argument("--static-selection-rollouts", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument(
        "--metrics-sort-score",
        choices=[
            "oracle_loss_mean",
            "oracle_loss_macro_subtype_event",
            "oracle_loss_macro_subtype_event_staticnorm",
        ],
        default="oracle_loss_mean",
    )
    parser.add_argument("--eval-start-indices", nargs="*", type=int, default=None)
    parser.add_argument(
        "--eval-start-selection",
        choices=[
            "uniform",
            "event_fraction",
            "event_rich",
            "event_transport_rich",
            "subtype_balanced_rich",
            "subtype_balanced_transport_rich",
        ],
        default="uniform",
    )
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--eval-selection-stride", type=int, default=64)
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.08)
    parser.add_argument("--lambda-switch", type=float, default=0.002)
    parser.add_argument("--event-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--event-subtype-particle-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--event-subtype-flux-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--event-subtype-thermal-reward-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--reward-loss-normalization",
        choices=["none", "staticnorm_subtype"],
        default="none",
    )
    parser.add_argument(
        "--reward-proxy-mode",
        choices=["forecast", "forecast_gain", "aoi", "uncertainty", "coverage", "instant_error"],
        default="forecast",
    )
    parser.add_argument("--energy-account", action="store_true")
    parser.add_argument("--energy-capacity", type=float, default=0.0)
    parser.add_argument("--initial-energy", type=float, default=0.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.0)
    parser.add_argument("--reserve-energy", type=float, default=0.0)
    parser.add_argument("--lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--soc-soft-penalty-buffer", type=float, default=0.0)
    parser.add_argument("--lambda-soc-soft-penalty", type=float, default=0.0)
    parser.add_argument("--lambda-duty-balance", type=float, default=0.0)
    parser.add_argument("--duty-balance-low", type=float, default=0.05)
    parser.add_argument("--duty-balance-high", type=float, default=0.95)
    parser.add_argument("--duty-balance-grace-steps", type=int, default=64)
    parser.add_argument("--duty-score-feedback", type=float, default=0.0)
    parser.add_argument("--duty-score-target", type=float, default=0.40)
    parser.add_argument("--duty-hard-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--duty-hard-low", type=float, default=0.08)
    parser.add_argument("--duty-hard-high", type=float, default=0.92)
    parser.add_argument("--duty-hard-score", type=float, default=8.0)
    parser.add_argument("--min-dwell-steps", type=int, default=1)
    parser.add_argument("--include-agent-cycle-phase", action="store_true")
    parser.add_argument("--agent-cycle-period-steps", type=int, default=0)
    parser.add_argument("--agent-cycle-dwell-steps", type=int, default=1)
    parser.add_argument("--include-observable-regime-belief", action="store_true")
    parser.add_argument("--regime-belief-lookback", type=int, default=6)
    parser.add_argument("--agent-context-columns", nargs="*", default=None)
    parser.add_argument("--include-event-flag-in-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-alert-context-features", action="store_true")
    parser.add_argument("--alert-context-columns", nargs="*", default=None)
    parser.add_argument("--alert-context-threshold", type=float, default=0.5)
    parser.add_argument("--alert-context-trend-lookback", type=int, default=6)
    parser.add_argument(
        "--measurement-update-mode",
        choices=["direct", "variance_weighted"],
        default="direct",
    )
    parser.add_argument("--eval-duty-constrained-baselines", action="store_true")
    parser.add_argument("--baseline-duty-hard-low", type=float, default=None)
    parser.add_argument("--baseline-duty-hard-high", type=float, default=None)
    parser.add_argument("--baseline-duty-hard-score", type=float, default=None)
    parser.add_argument("--baseline-duty-score-feedback", type=float, default=None)
    parser.add_argument("--primary-eval-duty-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--target-scales", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-loss-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-particle-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-flux-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-thermal-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--required-sensors", nargs="*", default=None)
    parser.add_argument("--disable-coverage-groups", action="store_true")
    parser.add_argument("--max-active", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-rollout-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    control_source_dir = Path(args.control_source_run_dir).resolve() if args.control_source_run_dir else None
    if control_source_dir is not None and not control_source_dir.is_dir():
        raise FileNotFoundError(f"Control source run does not exist: {control_source_dir}")
    truth_path = (
        Path(args.truth_csv)
        if args.truth_csv
        else (control_source_dir / "truth_v31_split.csv" if control_source_dir is not None else out_dir / "truth_v31_split.csv")
    )
    ratios = tuple(float(value) for value in args.split_ratios)
    bounds = partition_bounds(int(args.truth_steps), ratios)
    horizon = 8
    prior_starts = non_overlapping_starts(
        bounds=bounds["rl_train"],
        window_steps=int(args.candidate_prior_steps),
        horizon=horizon,
        count=int(args.candidate_prior_rollouts),
        seed=int(args.seed) + 811,
    )
    validation_starts = non_overlapping_starts(
        bounds=bounds["validation"],
        window_steps=int(args.static_selection_steps),
        horizon=horizon,
        count=int(args.static_selection_rollouts),
        seed=int(args.seed) + 1313,
    )
    eval_window_summary: dict[str, float | int] | None = None
    if args.eval_start_indices:
        final_starts = tuple(int(value) for value in args.eval_start_indices)
        final_selection = "manual_eval_start_indices"
    elif str(args.eval_start_selection) == "uniform":
        final_starts = non_overlapping_starts(
            bounds=bounds["final_test"],
            window_steps=int(args.eval_steps),
            horizon=horizon,
            count=int(args.eval_rollouts),
            seed=int(args.seed) + 1777,
        )
        final_selection = "uniform_random_non_overlapping_without_event_filtering"
    elif str(args.eval_start_selection) == "event_fraction":
        truth_path = ensure_truth(args, truth_path)
        truth = pd.read_csv(truth_path)
        final_starts = event_fraction_starts(
            truth,
            bounds=bounds["final_test"],
            window_steps=int(args.eval_steps),
            horizon=horizon,
            count=int(args.eval_rollouts),
            event_fraction=float(args.eval_event_fraction),
            stride=int(args.eval_selection_stride),
            seed=int(args.seed) + 1777,
        )
        final_selection = "event_fraction"
        eval_window_summary = summarize_eval_windows(
            truth,
            starts=final_starts,
            window_steps=int(args.eval_steps),
        )
    elif str(args.eval_start_selection).startswith("subtype_balanced"):
        truth_path = ensure_truth(args, truth_path)
        truth = pd.read_csv(truth_path)
        final_starts = subtype_balanced_non_overlapping_starts(
            truth,
            bounds=bounds["final_test"],
            window_steps=int(args.eval_steps),
            horizon=horizon,
            count=int(args.eval_rollouts),
            selection=str(args.eval_start_selection),
            stride=int(args.eval_selection_stride),
            seed=int(args.seed) + 1777,
        )
        final_selection = str(args.eval_start_selection)
        eval_window_summary = summarize_eval_windows(
            truth,
            starts=final_starts,
            window_steps=int(args.eval_steps),
        )
    else:
        truth_path = ensure_truth(args, truth_path)
        truth = pd.read_csv(truth_path)
        final_starts = rich_non_overlapping_starts(
            truth,
            bounds=bounds["final_test"],
            window_steps=int(args.eval_steps),
            horizon=horizon,
            count=int(args.eval_rollouts),
            selection=str(args.eval_start_selection),
            stride=int(args.eval_selection_stride),
            seed=int(args.seed) + 1777,
        )
        final_selection = str(args.eval_start_selection)
        eval_window_summary = summarize_eval_windows(
            truth,
            starts=final_starts,
            window_steps=int(args.eval_steps),
        )
    train_min, train_end = bounds["rl_train"]
    train_max = train_end - int(args.train_episode_len) - horizon - 1
    if train_max < train_min:
        raise ValueError("RL training partition is too short for the configured episode length")

    manifest = {
        "protocol": "chronological_split_v31_retraining",
        "control_source_run_dir": "" if control_source_dir is None else str(control_source_dir),
        "matched_control_assets": control_source_dir is not None,
        "truth_steps": int(args.truth_steps),
        "lookback": int(args.lookback),
        "seed": int(args.seed),
        "policy_seed": int(args.seed if args.policy_seed is None else args.policy_seed),
        "budget": float(args.budget),
        "target_weights": None if args.target_weights is None else [float(x) for x in args.target_weights],
        "target_scales": None if args.target_scales is None else [float(x) for x in args.target_scales],
        "subtype_loss_weighting": bool(args.subtype_loss_weighting),
        "subtype_particle_target_weights": None
        if args.subtype_particle_target_weights is None
        else [float(x) for x in args.subtype_particle_target_weights],
        "subtype_flux_target_weights": None
        if args.subtype_flux_target_weights is None
        else [float(x) for x in args.subtype_flux_target_weights],
        "subtype_thermal_target_weights": None
        if args.subtype_thermal_target_weights is None
        else [float(x) for x in args.subtype_thermal_target_weights],
        "split_ratios": list(ratios),
        "partitions": {name: [int(start), int(end)] for name, (start, end) in bounds.items()},
        "truth_event_design": {
            "event_coverage": float(args.event_coverage),
            "min_duration": int(args.min_duration),
            "max_duration": int(args.max_duration),
            "min_gap": int(args.min_gap),
            "lead_steps": int(args.lead_steps),
            "wind_margin_ms": float(args.wind_margin_ms),
            "cred_hysteresis_on": float(args.cred_hysteresis_on),
            "cred_hysteresis_off": float(args.cred_hysteresis_off),
            "flux_wind_exponent": float(args.flux_wind_exponent),
            "event_microstructure_sigma": float(args.event_microstructure_sigma),
            "event_microstructure_alpha": float(args.event_microstructure_alpha),
            "event_microstructure_diameter_scale": float(args.event_microstructure_diameter_scale),
            "event_microstructure_velocity_scale": float(args.event_microstructure_velocity_scale),
            "event_particle_microstructure_correlation": float(args.event_particle_microstructure_correlation),
            "event_subtypes_enabled": bool(args.event_subtypes_enabled),
            "event_subtype_particle_min_parsivel_availability": float(
                args.event_subtype_particle_min_parsivel_availability
            ),
            "event_subtype_particle_prob": float(args.event_subtype_particle_prob),
            "event_subtype_flux_prob": float(args.event_subtype_flux_prob),
            "event_subtype_thermal_prob": float(args.event_subtype_thermal_prob),
            "event_subtype_particle_flux_multiplier": float(args.event_subtype_particle_flux_multiplier),
            "event_subtype_flux_multiplier": float(args.event_subtype_flux_multiplier),
            "event_subtype_thermal_flux_multiplier": float(args.event_subtype_thermal_flux_multiplier),
            "event_subtype_particle_diameter_shift_mm": float(args.event_subtype_particle_diameter_shift_mm),
            "event_subtype_particle_velocity_boost_ms": float(args.event_subtype_particle_velocity_boost_ms),
            "event_subtype_flux_diameter_shift_mm": float(args.event_subtype_flux_diameter_shift_mm),
            "event_subtype_flux_velocity_boost_ms": float(args.event_subtype_flux_velocity_boost_ms),
            "event_subtype_thermal_surface_drop_c": float(args.event_subtype_thermal_surface_drop_c),
            "event_subtype_particle_humidity_boost_pct": float(args.event_subtype_particle_humidity_boost_pct),
            "event_subtype_flux_wind_boost_ms": float(args.event_subtype_flux_wind_boost_ms),
            "event_subtype_thermal_air_temp_drop_c": float(args.event_subtype_thermal_air_temp_drop_c),
            "event_subtype_latent_alpha": float(args.event_subtype_latent_alpha),
            "event_subtype_particle_latent_diameter_scale_mm": float(args.event_subtype_particle_latent_diameter_scale_mm),
            "event_subtype_particle_latent_velocity_scale_ms": float(args.event_subtype_particle_latent_velocity_scale_ms),
            "event_subtype_flux_latent_sigma": float(args.event_subtype_flux_latent_sigma),
            "event_subtype_flux_latent_linear_scale": float(args.event_subtype_flux_latent_linear_scale),
            "event_subtype_flux_latent_linear_offset": float(args.event_subtype_flux_latent_linear_offset),
            "event_subtype_flux_latent_linear_clip": float(args.event_subtype_flux_latent_linear_clip),
            "event_subtype_thermal_latent_surface_scale_c": float(args.event_subtype_thermal_latent_surface_scale_c),
            "event_subtype_latent_target_lag_steps": int(args.event_subtype_latent_target_lag_steps),
            "event_subtype_context_lead_steps": int(args.event_subtype_context_lead_steps),
            "event_subtype_context_noise_std": float(args.event_subtype_context_noise_std),
            "event_subtype_context_latent_strength": float(args.event_subtype_context_latent_strength),
        },
        "agent_context_columns": [str(x) for x in (args.agent_context_columns or ())],
        "primary_eval_duty_guard": bool(args.primary_eval_duty_guard),
        "oracle_pretrain": {"range": list(bounds["oracle_pretrain"])},
        "rl_train": {
            "range": list(bounds["rl_train"]),
            "ppo_start_min": int(train_min),
            "ppo_start_max": int(train_max),
            "normalization_range": list(bounds["rl_train"]),
            "candidate_prior_starts": list(prior_starts),
            "candidate_prior_steps": int(args.candidate_prior_steps),
        },
        "validation": {
            "static_selection_starts": list(validation_starts),
            "static_selection_steps": int(args.static_selection_steps),
            "static_selection_score": str(args.static_selection_score),
        },
        "final_test": {
            "eval_starts": list(final_starts),
            "eval_steps": int(args.eval_steps),
            "metrics_sort_score": str(args.metrics_sort_score),
            "selection": final_selection,
            "event_fraction": float(args.eval_event_fraction),
            "selection_stride": int(args.eval_selection_stride),
            "window_summary": eval_window_summary,
            "skip_rollout_evaluation": bool(args.skip_rollout_evaluation),
        },
        "operational_baselines": {
            "eval_duty_constrained_baselines": bool(args.eval_duty_constrained_baselines),
            "baseline_duty_hard_low": args.baseline_duty_hard_low,
            "baseline_duty_hard_high": args.baseline_duty_hard_high,
            "baseline_duty_hard_score": args.baseline_duty_hard_score,
            "baseline_duty_score_feedback": args.baseline_duty_score_feedback,
        },
        "reward_shaping": {
            "lambda_warmup_abort": float(args.lambda_warmup_abort),
            "lambda_switch": float(args.lambda_switch),
            "event_reward_multiplier": float(args.event_reward_multiplier),
            "reward_loss_normalization": str(args.reward_loss_normalization),
            "energy_account": bool(args.energy_account),
            "energy_capacity": float(args.energy_capacity),
            "initial_energy": float(args.initial_energy),
            "harvest_per_step": float(args.harvest_per_step),
            "reserve_energy": float(args.reserve_energy),
            "lambda_energy_deficit": float(args.lambda_energy_deficit),
            "soc_soft_penalty_buffer": float(args.soc_soft_penalty_buffer),
            "lambda_soc_soft_penalty": float(args.lambda_soc_soft_penalty),
            "lambda_duty_balance": float(args.lambda_duty_balance),
            "duty_balance_low": float(args.duty_balance_low),
            "duty_balance_high": float(args.duty_balance_high),
            "duty_balance_grace_steps": int(args.duty_balance_grace_steps),
            "duty_score_feedback": float(args.duty_score_feedback),
            "duty_score_target": float(args.duty_score_target),
            "duty_hard_guard": bool(args.duty_hard_guard),
            "duty_hard_low": float(args.duty_hard_low),
            "duty_hard_high": float(args.duty_hard_high),
            "duty_hard_score": float(args.duty_hard_score),
            "min_dwell_steps": int(max(1, int(args.min_dwell_steps))),
            "include_agent_cycle_phase": bool(args.include_agent_cycle_phase),
            "agent_cycle_period_steps": int(args.agent_cycle_period_steps),
            "agent_cycle_dwell_steps": max(1, int(args.agent_cycle_dwell_steps)),
            "include_observable_regime_belief": bool(args.include_observable_regime_belief),
            "regime_belief_lookback": max(1, int(args.regime_belief_lookback)),
            "include_event_flag_in_state": bool(args.include_event_flag_in_state),
            "actor_critic_event_context_source": "online_alert_proxy",
            "truth_event_labels_used_online": False,
            "include_alert_context_features": bool(args.include_alert_context_features),
            "alert_context_columns": [str(x) for x in (args.alert_context_columns or ())],
            "alert_context_threshold": float(args.alert_context_threshold),
            "alert_context_trend_lookback": max(1, int(args.alert_context_trend_lookback)),
        },
        "ppo_controls": {
            "ent_coef": float(args.ent_coef),
            "channel_marginal_entropy_coef": float(args.channel_marginal_entropy_coef),
            "policy_checkpoint_source": str(args.policy_checkpoint_source or ""),
            "evaluation_policy_mode": str(args.evaluation_policy_mode),
            "evaluation_sampling_seed": args.evaluation_sampling_seed,
            "evaluation_sampling_temperature": float(args.evaluation_sampling_temperature),
            "evaluation_temperature_candidates": [
                float(x) for x in (args.evaluation_temperature_candidates or ())
            ],
            "awbc_coef": float(args.awbc_coef),
            "awbc_decay_timesteps": max(0, int(args.awbc_decay_timesteps)),
            "awbc_label_stride": int(args.awbc_label_stride),
            "awbc_event_only": bool(args.awbc_event_only),
            "checkpoint_selection_interval_updates": max(
                0, int(args.checkpoint_selection_interval_updates)
            ),
            "checkpoint_selection_score": str(args.checkpoint_selection_score),
            "bc_pretrain_steps": int(args.bc_pretrain_steps),
            "bc_pretrain_epochs": int(args.bc_pretrain_epochs),
            "bc_pretrain_batch_size": int(args.bc_pretrain_batch_size),
            "bc_pretrain_loss_coef": float(args.bc_pretrain_loss_coef),
            "bc_pretrain_target_mode": str(args.bc_pretrain_target_mode),
            "bc_soft_temperature": float(args.bc_soft_temperature),
            "awbc_teacher_mode": str(args.awbc_teacher_mode),
            "awbc_teacher_alert_threshold": float(args.awbc_teacher_alert_threshold),
            "subtype_aux_coef": float(args.subtype_aux_coef),
            "subtype_aux_classes": int(args.subtype_aux_classes),
            "subtype_aux_lookahead_steps": int(args.subtype_aux_lookahead_steps),
            "subtype_action_ce_coef": float(args.subtype_action_ce_coef),
            "subtype_action_supervision_mode": str(args.subtype_action_supervision_mode),
            "subtype_action_event_only": bool(args.subtype_action_event_only),
            "subtype_router": bool(args.subtype_router),
            "subtype_router_min_confidence": float(args.subtype_router_min_confidence),
            "subtype_router_low_confidence_action": int(args.subtype_router_low_confidence_action),
            "temporal_encoder": bool(args.temporal_encoder),
            "temporal_hidden_dim": max(1, int(args.temporal_hidden_dim)),
            "awbc_teacher_calm_sensors": list(args.awbc_teacher_calm_sensors or ()),
            "awbc_teacher_event_sensors": list(args.awbc_teacher_event_sensors or ()),
            "awbc_teacher_subtype_calm_sensors": list(args.awbc_teacher_subtype_calm_sensors or ()),
            "awbc_teacher_subtype_particle_sensors": list(args.awbc_teacher_subtype_particle_sensors or ()),
            "awbc_teacher_subtype_flux_sensors": list(args.awbc_teacher_subtype_flux_sensors or ()),
            "awbc_teacher_subtype_thermal_sensors": list(args.awbc_teacher_subtype_thermal_sensors or ()),
            "awbc_teacher_auto_score_mode": str(args.awbc_teacher_auto_score_mode),
            "awbc_teacher_calm_pool_spec": str(args.awbc_teacher_calm_pool_spec or ""),
            "awbc_teacher_event_pool_spec": str(args.awbc_teacher_event_pool_spec or ""),
            "awbc_teacher_event_lookahead_steps": int(args.awbc_teacher_event_lookahead_steps),
            "awbc_teacher_energy_mpc_horizon": int(args.awbc_teacher_energy_mpc_horizon),
            "awbc_teacher_energy_mpc_soc_bins": int(args.awbc_teacher_energy_mpc_soc_bins),
            "awbc_teacher_energy_mpc_low_soc_ratio": float(args.awbc_teacher_energy_mpc_low_soc_ratio),
            "awbc_teacher_energy_mpc_high_soc_ratio": float(args.awbc_teacher_energy_mpc_high_soc_ratio),
            "awbc_teacher_energy_mpc_terminal_soc_weight": float(args.awbc_teacher_energy_mpc_terminal_soc_weight),
            "awbc_teacher_energy_mpc_max_actions": int(args.awbc_teacher_energy_mpc_max_actions),
            "awbc_teacher_energy_mpc_low_power_action": int(args.awbc_teacher_energy_mpc_low_power_action),
            "awbc_teacher_dwell_steps": int(args.awbc_teacher_dwell_steps),
            "prior_kl_coef": float(args.prior_kl_coef),
            "greedy_lookahead_steps": int(args.greedy_lookahead_steps),
            "event_start_prob": float(args.event_start_prob),
            "event_aware_critic": bool(args.event_aware_critic),
            "trainable_action_prior": bool(args.trainable_action_prior),
            "event_gated_actor": bool(args.event_gated_actor),
            "context_encoder": bool(args.context_encoder),
            "context_feature_dim": int(args.context_feature_dim),
            "context_hidden_dim": int(args.context_hidden_dim),
            "context_fusion_mode": str(args.context_fusion_mode),
            "context_layer_norm": bool(args.context_layer_norm),
            "soc_aux_horizon": int(args.soc_aux_horizon),
            "soc_aux_coef": float(args.soc_aux_coef),
            "learning_rate": float(args.learning_rate),
            "use_candidate_prior": bool(args.use_candidate_prior),
            "candidate_prior_scale": float(args.candidate_prior_scale),
            "oracle_candidate_mask_repeat": int(args.oracle_candidate_mask_repeat),
            "oracle_full_open_repeat": int(args.oracle_full_open_repeat),
            "oracle_candidate_mask_limit": int(args.oracle_candidate_mask_limit),
            "oracle_subtype_teacher_repeat": int(args.oracle_subtype_teacher_repeat),
            "oracle_subtype_teacher_lookahead_steps": int(args.oracle_subtype_teacher_lookahead_steps),
            "oracle_subtype_teacher_sensors": {
                "calm": list(args.oracle_subtype_teacher_calm_sensors or ()),
                "particle": list(args.oracle_subtype_teacher_particle_sensors or ()),
                "flux": list(args.oracle_subtype_teacher_flux_sensors or ()),
                "thermal": list(args.oracle_subtype_teacher_thermal_sensors or ()),
            },
            "oracle_loss_clip": float(args.oracle_loss_clip),
            "reward_proxy_mode": str(args.reward_proxy_mode),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "25_v2_train_custom_ppo.py"),
        "--output-dir",
        str(out_dir),
        "--truth-csv",
        str(truth_path),
        "--antaws-root",
        str(args.antaws_root),
        "--stations",
        *[str(station) for station in args.stations],
        "--sensor-cfg",
        str(args.sensor_cfg),
        "--seed",
        str(int(args.seed)),
        "--per-step-budget",
        str(float(args.budget)),
        "--startup-peak-budget",
        str(float(args.startup_peak_budget)),
        "--truth-steps",
        str(int(args.truth_steps)),
        "--freq-s",
        str(int(args.freq_s)),
        "--lookback",
        str(int(args.lookback)),
        "--blowing-snow-event-coverage",
        str(float(args.event_coverage)),
        "--blowing-snow-event-model",
        "semi_markov",
        "--blowing-snow-min-duration-steps",
        str(int(args.min_duration)),
        "--blowing-snow-max-duration-steps",
        str(int(args.max_duration)),
        "--blowing-snow-min-gap-steps",
        str(int(args.min_gap)),
        "--blowing-snow-lead-steps",
        str(int(args.lead_steps)),
        "--blowing-snow-wind-margin-ms",
        str(float(args.wind_margin_ms)),
        "--cred-hysteresis-on",
        str(float(args.cred_hysteresis_on)),
        "--cred-hysteresis-off",
        str(float(args.cred_hysteresis_off)),
        "--flux-wind-exponent",
        str(float(args.flux_wind_exponent)),
        "--event-microstructure-sigma",
        str(float(args.event_microstructure_sigma)),
        "--event-microstructure-alpha",
        str(float(args.event_microstructure_alpha)),
        "--event-microstructure-diameter-scale",
        str(float(args.event_microstructure_diameter_scale)),
        "--event-microstructure-velocity-scale",
        str(float(args.event_microstructure_velocity_scale)),
        "--event-particle-microstructure-correlation",
        str(float(args.event_particle_microstructure_correlation)),
        "--event-subtype-assignment",
        str(args.event_subtype_assignment),
        "--event-subtype-particle-min-parsivel-availability",
        str(float(args.event_subtype_particle_min_parsivel_availability)),
        "--event-subtype-particle-prob",
        str(float(args.event_subtype_particle_prob)),
        "--event-subtype-flux-prob",
        str(float(args.event_subtype_flux_prob)),
        "--event-subtype-thermal-prob",
        str(float(args.event_subtype_thermal_prob)),
        "--event-subtype-particle-flux-multiplier",
        str(float(args.event_subtype_particle_flux_multiplier)),
        "--event-subtype-flux-multiplier",
        str(float(args.event_subtype_flux_multiplier)),
        "--event-subtype-thermal-flux-multiplier",
        str(float(args.event_subtype_thermal_flux_multiplier)),
        "--event-subtype-particle-diameter-shift-mm",
        str(float(args.event_subtype_particle_diameter_shift_mm)),
        "--event-subtype-particle-velocity-boost-ms",
        str(float(args.event_subtype_particle_velocity_boost_ms)),
        "--event-subtype-flux-diameter-shift-mm",
        str(float(args.event_subtype_flux_diameter_shift_mm)),
        "--event-subtype-flux-velocity-boost-ms",
        str(float(args.event_subtype_flux_velocity_boost_ms)),
        "--event-subtype-thermal-surface-drop-c",
        str(float(args.event_subtype_thermal_surface_drop_c)),
        "--event-subtype-particle-humidity-boost-pct",
        str(float(args.event_subtype_particle_humidity_boost_pct)),
        "--event-subtype-flux-wind-boost-ms",
        str(float(args.event_subtype_flux_wind_boost_ms)),
        "--event-subtype-thermal-air-temp-drop-c",
        str(float(args.event_subtype_thermal_air_temp_drop_c)),
        "--event-subtype-latent-alpha",
        str(float(args.event_subtype_latent_alpha)),
        "--event-subtype-particle-latent-diameter-scale-mm",
        str(float(args.event_subtype_particle_latent_diameter_scale_mm)),
        "--event-subtype-particle-latent-velocity-scale-ms",
        str(float(args.event_subtype_particle_latent_velocity_scale_ms)),
        "--event-subtype-flux-latent-sigma",
        str(float(args.event_subtype_flux_latent_sigma)),
        "--event-subtype-flux-latent-linear-scale",
        str(float(args.event_subtype_flux_latent_linear_scale)),
        "--event-subtype-flux-latent-linear-offset",
        str(float(args.event_subtype_flux_latent_linear_offset)),
        "--event-subtype-flux-latent-linear-clip",
        str(float(args.event_subtype_flux_latent_linear_clip)),
        "--event-subtype-thermal-latent-surface-scale-c",
        str(float(args.event_subtype_thermal_latent_surface_scale_c)),
        "--event-subtype-latent-target-lag-steps",
        str(int(args.event_subtype_latent_target_lag_steps)),
        "--event-subtype-context-lead-steps",
        str(int(args.event_subtype_context_lead_steps)),
        "--event-subtype-context-noise-std",
        str(float(args.event_subtype_context_noise_std)),
        "--event-subtype-context-latent-strength",
        str(float(args.event_subtype_context_latent_strength)),
        "--oracle-type",
        str(args.oracle_type),
        "--oracle-rollout-steps",
        str(int(args.oracle_rollout_steps)),
        "--oracle-rollouts-per-policy",
        str(int(args.oracle_rollouts_per_policy)),
        "--oracle-full-open-repeat",
        str(int(args.oracle_full_open_repeat)),
        "--oracle-epochs",
        str(int(args.oracle_epochs)),
        "--oracle-batch-size",
        str(int(args.oracle_batch_size)),
        "--oracle-loss-clip",
        str(float(args.oracle_loss_clip)),
        "--oracle-candidate-mask-repeat",
        str(int(args.oracle_candidate_mask_repeat)),
        "--oracle-candidate-mask-limit",
        str(int(args.oracle_candidate_mask_limit)),
        "--oracle-subtype-teacher-repeat",
        str(int(args.oracle_subtype_teacher_repeat)),
        "--oracle-subtype-teacher-lookahead-steps",
        str(int(args.oracle_subtype_teacher_lookahead_steps)),
        "--oracle-device",
        str(args.oracle_device),
        "--oracle-inference-device",
        str(args.oracle_inference_device),
        "--oracle-start-idx",
        str(bounds["oracle_pretrain"][0]),
        "--oracle-end-idx",
        str(bounds["oracle_pretrain"][1]),
        "--normalization-start-idx",
        str(bounds["rl_train"][0]),
        "--normalization-end-idx",
        str(bounds["rl_train"][1]),
        "--train-start-min",
        str(train_min),
        "--train-start-max",
        str(train_max),
        "--train-episode-len",
        str(int(args.train_episode_len)),
        "--total-timesteps",
        str(int(args.total_timesteps)),
        "--n-steps",
        str(int(args.n_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--ent-coef",
        str(float(args.ent_coef)),
        "--channel-marginal-entropy-coef",
        str(float(args.channel_marginal_entropy_coef)),
        "--awbc-coef",
        str(float(args.awbc_coef)),
        "--awbc-decay-timesteps",
        str(max(0, int(args.awbc_decay_timesteps))),
        "--awbc-label-stride",
        str(int(args.awbc_label_stride)),
        "--awbc-event-only" if bool(args.awbc_event_only) else "--no-awbc-event-only",
        "--checkpoint-selection-interval-updates",
        str(max(0, int(args.checkpoint_selection_interval_updates))),
        "--checkpoint-selection-score",
        str(args.checkpoint_selection_score),
        "--bc-pretrain-steps",
        str(int(args.bc_pretrain_steps)),
        "--bc-pretrain-epochs",
        str(int(args.bc_pretrain_epochs)),
        "--bc-pretrain-batch-size",
        str(int(args.bc_pretrain_batch_size)),
        "--bc-pretrain-loss-coef",
        str(float(args.bc_pretrain_loss_coef)),
        "--bc-pretrain-target-mode",
        str(args.bc_pretrain_target_mode),
        "--bc-soft-temperature",
        str(float(args.bc_soft_temperature)),
        "--subtype-aux-coef",
        str(float(args.subtype_aux_coef)),
        "--subtype-aux-classes",
        str(max(2, int(args.subtype_aux_classes))),
        "--subtype-aux-lookahead-steps",
        str(max(0, int(args.subtype_aux_lookahead_steps))),
        "--subtype-action-ce-coef",
        str(float(args.subtype_action_ce_coef)),
        "--subtype-action-supervision-mode",
        str(args.subtype_action_supervision_mode),
        "--subtype-action-event-only" if bool(args.subtype_action_event_only) else "--no-subtype-action-event-only",
        "--subtype-action-margin-coef",
        str(float(args.subtype_action_margin_coef)),
        "--subtype-action-margin",
        str(float(args.subtype_action_margin)),
        "--subtype-router" if bool(args.subtype_router) else "--no-subtype-router",
        "--subtype-router-min-confidence",
        str(float(args.subtype_router_min_confidence)),
        "--subtype-router-low-confidence-action",
        str(int(args.subtype_router_low_confidence_action)),
        "--awbc-teacher-mode",
        str(args.awbc_teacher_mode),
        "--awbc-teacher-event-lookahead-steps",
        str(int(args.awbc_teacher_event_lookahead_steps)),
        "--awbc-teacher-alert-threshold",
        str(float(args.awbc_teacher_alert_threshold)),
        "--awbc-teacher-energy-mpc-horizon",
        str(int(args.awbc_teacher_energy_mpc_horizon)),
        "--awbc-teacher-energy-mpc-soc-bins",
        str(int(args.awbc_teacher_energy_mpc_soc_bins)),
        "--awbc-teacher-energy-mpc-low-soc-ratio",
        str(float(args.awbc_teacher_energy_mpc_low_soc_ratio)),
        "--awbc-teacher-energy-mpc-high-soc-ratio",
        str(float(args.awbc_teacher_energy_mpc_high_soc_ratio)),
        "--awbc-teacher-energy-mpc-terminal-soc-weight",
        str(float(args.awbc_teacher_energy_mpc_terminal_soc_weight)),
        "--awbc-teacher-energy-mpc-max-actions",
        str(int(args.awbc_teacher_energy_mpc_max_actions)),
        "--awbc-teacher-energy-mpc-low-power-action",
        str(int(args.awbc_teacher_energy_mpc_low_power_action)),
        "--awbc-teacher-dwell-steps",
        str(int(args.awbc_teacher_dwell_steps)),
        "--agent-cycle-period-steps",
        str(int(args.agent_cycle_period_steps)),
        "--agent-cycle-dwell-steps",
        str(int(args.agent_cycle_dwell_steps)),
        "--regime-belief-lookback",
        str(max(1, int(args.regime_belief_lookback))),
        "--prior-kl-coef",
        str(float(args.prior_kl_coef)),
        "--greedy-lookahead-steps",
        str(int(args.greedy_lookahead_steps)),
        "--device",
        str(args.device),
        "--candidate-prior-steps",
        str(int(args.candidate_prior_steps)),
        "--candidate-prior-rollouts",
        str(int(args.candidate_prior_rollouts)),
        "--candidate-prior-scale",
        str(float(args.candidate_prior_scale)),
        "--candidate-prior-start-indices",
        *[str(value) for value in prior_starts],
        "--static-selection-steps",
        str(int(args.static_selection_steps)),
        "--static-selection-score",
        str(args.static_selection_score),
        "--static-selection-start-indices",
        *[str(value) for value in validation_starts],
        "--eval-steps",
        str(int(args.eval_steps)),
        "--eval-rollouts",
        str(int(args.eval_rollouts)),
        "--metrics-sort-score",
        str(args.metrics_sort_score),
        "--eval-event-fraction",
        str(float(args.eval_event_fraction)),
        "--lambda-warmup-abort",
        str(float(args.lambda_warmup_abort)),
        "--lambda-switch",
        str(float(args.lambda_switch)),
        "--event-reward-multiplier",
        str(float(args.event_reward_multiplier)),
        "--event-subtype-particle-reward-multiplier",
        str(float(args.event_subtype_particle_reward_multiplier)),
        "--event-subtype-flux-reward-multiplier",
        str(float(args.event_subtype_flux_reward_multiplier)),
        "--event-subtype-thermal-reward-multiplier",
        str(float(args.event_subtype_thermal_reward_multiplier)),
        "--reward-loss-normalization",
        str(args.reward_loss_normalization),
        "--reward-proxy-mode",
        str(args.reward_proxy_mode),
        "--lambda-energy-deficit",
        str(float(args.lambda_energy_deficit)),
        "--soc-soft-penalty-buffer",
        str(float(args.soc_soft_penalty_buffer)),
        "--lambda-soc-soft-penalty",
        str(float(args.lambda_soc_soft_penalty)),
        "--lambda-duty-balance",
        str(float(args.lambda_duty_balance)),
        "--duty-balance-low",
        str(float(args.duty_balance_low)),
        "--duty-balance-high",
        str(float(args.duty_balance_high)),
        "--duty-balance-grace-steps",
        str(int(args.duty_balance_grace_steps)),
        "--duty-score-feedback",
        str(float(args.duty_score_feedback)),
        "--duty-score-target",
        str(float(args.duty_score_target)),
        "--duty-hard-low",
        str(float(args.duty_hard_low)),
        "--duty-hard-high",
        str(float(args.duty_hard_high)),
        "--duty-hard-score",
        str(float(args.duty_hard_score)),
        "--min-dwell-steps",
        str(int(args.min_dwell_steps)),
        "--eval-start-indices",
        *[str(value) for value in final_starts],
        "--event-start-prob",
        str(float(args.event_start_prob)),
        "--soc-aux-horizon",
        str(int(args.soc_aux_horizon)),
        "--soc-aux-coef",
        str(float(args.soc_aux_coef)),
    ]
    if args.policy_seed is not None:
        cmd.extend(["--policy-seed", str(int(args.policy_seed))])
    if control_source_dir is not None:
        cmd.extend(["--control-source-run-dir", str(control_source_dir)])
    if bool(args.validate_control_source_only):
        cmd.append("--validate-control-source-only")
    if bool(args.event_subtypes_enabled):
        cmd.append("--event-subtypes-enabled")
    cmd.append("--event-aware-critic" if bool(args.event_aware_critic) else "--no-event-aware-critic")
    cmd.append(
        "--trainable-action-prior"
        if bool(args.trainable_action_prior)
        else "--no-trainable-action-prior"
    )
    cmd.append(
        "--nonlinear-action-embedding"
        if bool(args.nonlinear_action_embedding)
        else "--no-nonlinear-action-embedding"
    )
    cmd.append("--event-gated-actor" if bool(args.event_gated_actor) else "--no-event-gated-actor")
    cmd.append("--context-encoder" if bool(args.context_encoder) else "--no-context-encoder")
    cmd.extend(["--context-feature-dim", str(max(0, int(args.context_feature_dim)))])
    cmd.extend(["--context-hidden-dim", str(max(1, int(args.context_hidden_dim)))])
    cmd.extend(["--context-fusion-mode", str(args.context_fusion_mode)])
    cmd.append("--context-layer-norm" if bool(args.context_layer_norm) else "--no-context-layer-norm")
    cmd.append("--temporal-encoder" if bool(args.temporal_encoder) else "--no-temporal-encoder")
    cmd.extend(["--temporal-hidden-dim", str(max(1, int(args.temporal_hidden_dim)))])
    cmd.append(
        "--separate-actor-critic-grad-clip"
        if bool(args.separate_actor_critic_grad_clip)
        else "--no-separate-actor-critic-grad-clip"
    )
    cmd.extend(["--learning-rate", str(float(args.learning_rate))])
    if bool(args.include_agent_cycle_phase):
        cmd.append("--include-agent-cycle-phase")
    if bool(args.include_observable_regime_belief):
        cmd.append("--include-observable-regime-belief")
    cmd.append(
        "--include-event-flag-in-state"
        if bool(args.include_event_flag_in_state)
        else "--no-include-event-flag-in-state"
    )
    if bool(args.include_alert_context_features):
        cmd.append("--include-alert-context-features")
    cmd.extend(["--alert-context-threshold", str(float(args.alert_context_threshold))])
    cmd.extend(["--alert-context-trend-lookback", str(max(1, int(args.alert_context_trend_lookback)))])
    cmd.extend(["--measurement-update-mode", str(args.measurement_update_mode)])
    if bool(args.duty_hard_guard):
        cmd.append("--duty-hard-guard")
    if bool(args.eval_duty_constrained_baselines):
        cmd.append("--eval-duty-constrained-baselines")
    append_option(
        cmd,
        "--oracle-subtype-teacher-calm-sensors",
        None
        if args.oracle_subtype_teacher_calm_sensors is None
        else [str(x) for x in args.oracle_subtype_teacher_calm_sensors],
    )
    append_option(
        cmd,
        "--oracle-subtype-teacher-particle-sensors",
        None
        if args.oracle_subtype_teacher_particle_sensors is None
        else [str(x) for x in args.oracle_subtype_teacher_particle_sensors],
    )
    append_option(
        cmd,
        "--oracle-subtype-teacher-flux-sensors",
        None
        if args.oracle_subtype_teacher_flux_sensors is None
        else [str(x) for x in args.oracle_subtype_teacher_flux_sensors],
    )
    append_option(
        cmd,
        "--oracle-subtype-teacher-thermal-sensors",
        None
        if args.oracle_subtype_teacher_thermal_sensors is None
        else [str(x) for x in args.oracle_subtype_teacher_thermal_sensors],
    )
    append_option(
        cmd,
        "--awbc-teacher-calm-sensors",
        None if args.awbc_teacher_calm_sensors is None else [str(x) for x in args.awbc_teacher_calm_sensors],
    )
    append_option(
        cmd,
        "--awbc-teacher-event-sensors",
        None if args.awbc_teacher_event_sensors is None else [str(x) for x in args.awbc_teacher_event_sensors],
    )
    append_option(
        cmd,
        "--awbc-teacher-calm-pool-spec",
        None if args.awbc_teacher_calm_pool_spec is None else [str(args.awbc_teacher_calm_pool_spec)],
    )
    append_option(
        cmd,
        "--awbc-teacher-event-pool-spec",
        None if args.awbc_teacher_event_pool_spec is None else [str(args.awbc_teacher_event_pool_spec)],
    )
    append_option(
        cmd,
        "--awbc-teacher-subtype-calm-sensors",
        None if args.awbc_teacher_subtype_calm_sensors is None else [str(x) for x in args.awbc_teacher_subtype_calm_sensors],
    )
    append_option(
        cmd,
        "--awbc-teacher-subtype-particle-sensors",
        None
        if args.awbc_teacher_subtype_particle_sensors is None
        else [str(x) for x in args.awbc_teacher_subtype_particle_sensors],
    )
    append_option(
        cmd,
        "--awbc-teacher-subtype-flux-sensors",
        None if args.awbc_teacher_subtype_flux_sensors is None else [str(x) for x in args.awbc_teacher_subtype_flux_sensors],
    )
    append_option(
        cmd,
        "--awbc-teacher-subtype-thermal-sensors",
        None
        if args.awbc_teacher_subtype_thermal_sensors is None
        else [str(x) for x in args.awbc_teacher_subtype_thermal_sensors],
    )
    cmd.extend(["--awbc-teacher-auto-score-mode", str(args.awbc_teacher_auto_score_mode)])
    append_option(
        cmd,
        "--agent-context-columns",
        None if args.agent_context_columns is None else [str(x) for x in args.agent_context_columns],
    )
    append_option(
        cmd,
        "--alert-context-columns",
        None if args.alert_context_columns is None else [str(x) for x in args.alert_context_columns],
    )
    if args.baseline_duty_hard_low is not None:
        cmd.extend(["--baseline-duty-hard-low", str(float(args.baseline_duty_hard_low))])
    if args.baseline_duty_hard_high is not None:
        cmd.extend(["--baseline-duty-hard-high", str(float(args.baseline_duty_hard_high))])
    if args.baseline_duty_hard_score is not None:
        cmd.extend(["--baseline-duty-hard-score", str(float(args.baseline_duty_hard_score))])
    if args.baseline_duty_score_feedback is not None:
        cmd.extend(["--baseline-duty-score-feedback", str(float(args.baseline_duty_score_feedback))])
    if bool(args.primary_eval_duty_guard):
        cmd.append("--primary-eval-duty-guard")
    if bool(args.use_candidate_prior):
        cmd.append("--use-oracle-candidate-prior")
    if bool(args.skip_rollout_evaluation):
        cmd.append("--skip-evaluation")
    if args.policy_checkpoint_source:
        cmd.extend(["--policy-checkpoint-source", str(args.policy_checkpoint_source)])
    cmd.extend(["--evaluation-policy-mode", str(args.evaluation_policy_mode)])
    if args.evaluation_sampling_seed is not None:
        cmd.extend(["--evaluation-sampling-seed", str(int(args.evaluation_sampling_seed))])
    cmd.extend(
        ["--evaluation-sampling-temperature", str(float(args.evaluation_sampling_temperature))]
    )
    append_option(
        cmd,
        "--evaluation-temperature-candidates",
        None
        if args.evaluation_temperature_candidates is None
        else [str(float(x)) for x in args.evaluation_temperature_candidates],
    )
    if bool(args.energy_account):
        cmd.extend(
            [
                "--energy-account",
                "--energy-capacity",
                str(float(args.energy_capacity)),
                "--initial-energy",
                str(float(args.initial_energy)),
                "--harvest-per-step",
                str(float(args.harvest_per_step)),
                "--reserve-energy",
                str(float(args.reserve_energy)),
            ]
        )
    cmd.append("--subtype-loss-weighting" if bool(args.subtype_loss_weighting) else "--no-subtype-loss-weighting")
    append_option(cmd, "--target-weights", None if args.target_weights is None else [str(float(x)) for x in args.target_weights])
    append_option(cmd, "--target-scales", None if args.target_scales is None else [str(float(x)) for x in args.target_scales])
    append_option(
        cmd,
        "--subtype-particle-target-weights",
        None if args.subtype_particle_target_weights is None else [str(float(x)) for x in args.subtype_particle_target_weights],
    )
    append_option(
        cmd,
        "--subtype-flux-target-weights",
        None if args.subtype_flux_target_weights is None else [str(float(x)) for x in args.subtype_flux_target_weights],
    )
    append_option(
        cmd,
        "--subtype-thermal-target-weights",
        None if args.subtype_thermal_target_weights is None else [str(float(x)) for x in args.subtype_thermal_target_weights],
    )
    append_option(cmd, "--required-sensors", None if args.required_sensors is None else [str(x) for x in args.required_sensors])
    if args.disable_coverage_groups:
        cmd.append("--disable-coverage-groups")
    if args.max_active is not None:
        cmd.extend(["--max-active", str(int(args.max_active))])

    if args.dry_run:
        print(manifest_path)
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)
    print(manifest_path)


if __name__ == "__main__":
    main()
