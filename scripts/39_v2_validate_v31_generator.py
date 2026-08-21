#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_sources.public_weather_synthesis import (  # noqa: E402
    PublicWeatherSynthesisConfig,
    _acf,
    _bool_runs,
    build_antaws_anchor,
    generate_public_weather_truth,
)


BASE_VARIABLES = [
    "air_temperature_c",
    "wind_speed_ms",
    "relative_humidity",
    "air_pressure_pa",
]


def _window_event_fractions(mask: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=float).reshape(-1)
    if arr.size < window:
        return np.asarray([float(np.mean(arr))], dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0), dtype=float)
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def _psd_cpd(values: np.ndarray, *, freq_s: int) -> tuple[np.ndarray, np.ndarray]:
    samples_per_day = 86400.0 / float(freq_s)
    freq, psd = signal.welch(
        np.asarray(values, dtype=float),
        fs=samples_per_day,
        nperseg=min(1024, len(values)),
        detrend="constant",
    )
    return freq, psd


def _log_psd_mse(reference: np.ndarray, synthetic: np.ndarray, *, freq_s: int) -> float:
    n = min(len(reference), len(synthetic))
    freq_ref, psd_ref = _psd_cpd(reference[:n], freq_s=freq_s)
    freq_syn, psd_syn = _psd_cpd(synthetic[:n], freq_s=freq_s)
    freq_grid = np.linspace(0.1, 4.0, 256)
    ref_interp = np.interp(freq_grid, freq_ref, psd_ref)
    syn_interp = np.interp(freq_grid, freq_syn, psd_syn)
    eps = 1e-12
    return float(np.mean((np.log10(ref_interp + eps) - np.log10(syn_interp + eps)) ** 2))


def _linear_slope_loglog(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0.0) & (y_arr > 0.0)
    if int(np.sum(mask)) < 4:
        return float("nan")
    slope, _intercept = np.polyfit(np.log(x_arr[mask]), np.log(y_arr[mask]), deg=1)
    return float(slope)


def _write_validation_plots(
    *,
    out_dir: Path,
    anchor: pd.DataFrame,
    synthetic: pd.DataFrame,
    event_fractions: np.ndarray,
    freq_s: int,
    max_lag: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(anchor), len(synthetic))
    anchor_wind = anchor["wind_speed_ms"].to_numpy(dtype=float)[:n]
    synthetic_wind = synthetic["wind_speed_ms"].to_numpy(dtype=float)[:n]

    lags = np.arange(1, max_lag + 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.plot(lags, _acf(anchor_wind, max_lag=max_lag), label="AntAWS anchor", linewidth=2)
    ax.plot(lags, _acf(synthetic_wind, max_lag=max_lag), label="V3.1 synthetic", linewidth=2)
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("ACF")
    ax.set_title("Wind-speed autocorrelation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "figure_g1_acf_comparison.png", dpi=300)
    fig.savefig(out_dir / "figure_g1_acf_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.hist(event_fractions, bins=np.linspace(0.0, 1.0, 31), color="#355C7D", alpha=0.85)
    ax.axvline(0.25, color="#2A9D8F", linestyle="--", linewidth=1.6, label="calm threshold")
    ax.axvline(0.75, color="#E76F51", linestyle="--", linewidth=1.6, label="event-heavy threshold")
    ax.set_xlabel("event fraction in 512-step windows")
    ax.set_ylabel("count")
    ax.set_title("Window-level event-fraction distribution")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "figure_g1_event_fraction_dist.png", dpi=300)
    fig.savefig(out_dir / "figure_g1_event_fraction_dist.pdf")
    plt.close(fig)

    freq_ref, psd_ref = _psd_cpd(anchor_wind, freq_s=freq_s)
    freq_syn, psd_syn = _psd_cpd(synthetic_wind, freq_s=freq_s)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.plot(freq_ref, psd_ref, label="AntAWS anchor", linewidth=2)
    ax.plot(freq_syn, psd_syn, label="V3.1 synthetic", linewidth=2)
    ax.set_xlim(0.1, 4.0)
    ax.set_yscale("log")
    ax.set_xlabel("frequency (cycles/day)")
    ax.set_ylabel("PSD")
    ax.set_title("Wind-speed PSD")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "figure_g1_psd_comparison.png", dpi=300)
    fig.savefig(out_dir / "figure_g1_psd_comparison.pdf")
    plt.close(fig)

    active = synthetic["event_flag"].to_numpy(dtype=bool)
    wind = synthetic["wind_speed_ms"].to_numpy(dtype=float)
    flux = synthetic["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float)
    diameter = synthetic["snow_particle_mean_diameter_mm"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)
    durations = [(end - start) * freq_s / 3600.0 for start, end in _bool_runs(active)]
    axes[0].hist(durations, bins=20, color="#6C5B7B", alpha=0.85)
    axes[0].axvline(15.0, color="black", linestyle="--", linewidth=1.3, label="15 h target")
    axes[0].set_xlabel("event duration (h)")
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False)

    plot_mask = active & (flux > 0.0)
    axes[1].scatter(wind[plot_mask], flux[plot_mask], s=8, alpha=0.35, color="#F67280")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("wind speed (m/s)")
    axes[1].set_ylabel("mass flux")
    axes[1].set_title("Flux-wind coupling")

    particle_mask = active & (diameter > 0.0)
    axes[2].scatter(wind[particle_mask], diameter[particle_mask], s=8, alpha=0.35, color="#2A9D8F")
    axes[2].set_xlabel("wind speed (m/s)")
    axes[2].set_ylabel("diameter (mm)")
    axes[2].set_title("Particle-size relation")
    fig.savefig(out_dir / "figure_g1_blowing_snow_stats.png", dpi=300)
    fig.savefig(out_dir / "figure_g1_blowing_snow_stats.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the V3.1 AntAWS-anchored generator.")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["D-17"])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--freq-s", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--event-coverage", type=float, default=0.24)
    parser.add_argument("--min-duration", type=int, default=12)
    parser.add_argument("--max-duration", type=int, default=24)
    parser.add_argument("--min-gap", type=int, default=4)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--out-dir", default="reports/v3_supplement_assets")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PublicWeatherSynthesisConfig(
        antaws_root=Path(args.antaws_root),
        stations=tuple(str(s) for s in args.stations),
        steps=int(args.steps),
        freq_s=int(args.freq_s),
        seed=int(args.seed),
        phase_keep_fraction=0.15,
        blowing_snow_event_coverage=float(args.event_coverage),
        blowing_snow_event_model="semi_markov",
        blowing_snow_min_duration_steps=int(args.min_duration),
        blowing_snow_max_duration_steps=int(args.max_duration),
        blowing_snow_min_gap_steps=int(args.min_gap),
        blowing_snow_lead_steps=6,
        blowing_snow_wind_margin_ms=1.2,
        flux_wind_exponent=3.0,
    )
    anchor = build_antaws_anchor(cfg.antaws_root, cfg.stations, freq_s=int(cfg.freq_s))
    synthetic, meta = generate_public_weather_truth(cfg)
    synthetic.to_csv(out_dir / "g1_v31_synthetic_truth.csv", index=False)
    (out_dir / "g1_v31_synthetic_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n = min(len(anchor), len(synthetic))
    anchor_wind = anchor["wind_speed_ms"].to_numpy(dtype=float)[:n]
    synthetic_wind = synthetic["wind_speed_ms"].to_numpy(dtype=float)[:n]
    max_lag = max(1, int(round(12 * 3600 / float(cfg.freq_s))))
    acf_delta = float(np.max(np.abs(_acf(anchor_wind, max_lag=max_lag) - _acf(synthetic_wind, max_lag=max_lag))))

    event_fractions = _window_event_fractions(synthetic["event_flag"].to_numpy(dtype=bool), int(args.window))
    p_event_heavy = float(np.mean(event_fractions > 0.75))
    p_calm = float(np.mean(event_fractions < 0.25))

    ks_values = []
    for col in BASE_VARIABLES:
        real = anchor[col].to_numpy(dtype=float)
        synth = synthetic[col].to_numpy(dtype=float)
        ks_values.append(float(stats.ks_2samp(real, synth).statistic))
    max_ks = float(np.max(ks_values))

    active = synthetic["event_flag"].to_numpy(dtype=bool)
    durations_h = np.asarray([(end - start) * int(cfg.freq_s) / 3600.0 for start, end in _bool_runs(active)])
    median_duration = float(np.median(durations_h)) if durations_h.size else float("nan")
    wind = synthetic["wind_speed_ms"].to_numpy(dtype=float)
    flux = synthetic["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float)
    diameter = synthetic["snow_particle_mean_diameter_mm"].to_numpy(dtype=float)
    event_flux_mask = active & (flux > 0.0) & (wind > 0.0)
    flux_wind_slope = _linear_slope_loglog(wind[event_flux_mask], flux[event_flux_mask])
    particle_mask = active & (diameter > 0.0)
    if int(np.sum(particle_mask)) >= 4:
        diameter_wind_rho = float(stats.spearmanr(wind[particle_mask], diameter[particle_mask]).correlation)
    else:
        diameter_wind_rho = float("nan")

    psd_log_mse = _log_psd_mse(anchor_wind, synthetic_wind, freq_s=int(cfg.freq_s))

    rows = [
        {
            "metric": "G1-V1 wind_speed_acf_max_abs_delta_lag1_12h",
            "value": acf_delta,
            "threshold": "<0.05",
            "passed": acf_delta < 0.05,
        },
        {
            "metric": "G1-V2 p_event_fraction_gt_0p75",
            "value": p_event_heavy,
            "threshold": ">0.05",
            "passed": p_event_heavy > 0.05,
        },
        {
            "metric": "G1-V2 p_event_fraction_lt_0p25",
            "value": p_calm,
            "threshold": ">0.30",
            "passed": p_calm > 0.30,
        },
        {
            "metric": "G1-V3a max_ks_antaws_scalar_base_variables",
            "value": max_ks,
            "threshold": "<0.05",
            "passed": max_ks < 0.05,
        },
        {
            "metric": "G1-V3b event_duration_median_h",
            "value": median_duration,
            "threshold": "12-20",
            "passed": 12.0 <= median_duration <= 20.0,
        },
        {
            "metric": "G1-V3b flux_wind_loglog_slope",
            "value": flux_wind_slope,
            "threshold": "2.5-3.5",
            "passed": 2.5 <= flux_wind_slope <= 3.5,
        },
        {
            "metric": "G1-V3b diameter_wind_spearman_rho",
            "value": diameter_wind_rho,
            "threshold": "<-0.30",
            "passed": (not math.isnan(diameter_wind_rho)) and diameter_wind_rho < -0.30,
        },
        {
            "metric": "G1-V4 wind_speed_psd_log_mse_0p1_4cpd",
            "value": psd_log_mse,
            "threshold": "<0.10",
            "passed": psd_log_mse < 0.10,
        },
    ]
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "exp_g1_generator_validation.csv", index=False)
    _write_validation_plots(
        out_dir=out_dir,
        anchor=anchor,
        synthetic=synthetic,
        event_fractions=event_fractions,
        freq_s=int(cfg.freq_s),
        max_lag=max_lag,
    )
    print(result.to_string(index=False))
    print(out_dir / "exp_g1_generator_validation.csv")


if __name__ == "__main__":
    main()
