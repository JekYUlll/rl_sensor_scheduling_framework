#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "paper" / "figures"))

from data_sources.public_weather_synthesis import build_antaws_anchor  # noqa: E402
from paper_plot_style import PALETTE, apply_paper_style as apply_common_style  # noqa: E402


PAPER = {
    "blue_fill": "#DDEBF7",
    "green_fill": "#E4F2E8",
    "yellow_fill": "#FFF4CC",
    "red_fill": "#FFE2DE",
    "cyan_fill": "#DDF2F0",
    "orange_fill": "#FFECD8",
    "gray_fill": "#EEF1F4",
    "blue": PALETTE["blue"],
    "green": PALETTE["teal"],
    "amber": PALETTE["orange"],
    "red": PALETTE["vermillion"],
    "slate": PALETTE["gray"],
    "ink": PALETTE["dark"],
    "grid": "#D8DCE0",
}


def apply_paper_style() -> None:
    apply_common_style(base_size=9.3)
    plt.rcParams.update({"axes.edgecolor": "#5C6368", "axes.linewidth": 0.65})


def soften_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=PAPER["grid"], linewidth=0.45, alpha=0.65)
    ax.tick_params(width=0.6, length=2.4, color="#5C6368")
    ax.title.set_fontweight("semibold")


def acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= max_lag + 1:
        return np.full(max_lag, np.nan)
    arr = arr - float(np.mean(arr))
    denom = float(np.dot(arr, arr))
    if denom <= 0.0:
        return np.full(max_lag, np.nan)
    return np.asarray([float(np.dot(arr[:-lag], arr[lag:]) / denom) for lag in range(1, max_lag + 1)])


def window_event_fractions(mask: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=float).reshape(-1)
    if arr.size < window:
        return np.asarray([float(np.mean(arr))], dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0), dtype=float)
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def parse_rule(rule: str) -> tuple[str, float | tuple[float, float]]:
    text = str(rule).strip()
    if text.startswith("<"):
        return "<", float(text[1:])
    if text.startswith(">"):
        return ">", float(text[1:])
    if "-" in text:
        lo, hi = text.split("-", 1)
        return "range", (float(lo), float(hi))
    raise ValueError(f"Unsupported rule: {rule}")


def pass_margin(value: float, rule: str) -> float:
    kind, threshold = parse_rule(rule)
    if kind == "<":
        t = float(threshold)
        if t < 0.0:
            return float(value) / t
        return t / max(float(value), 1e-12)
    if kind == ">":
        return float(value) / max(float(threshold), 1e-12)
    lo, hi = threshold
    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    distance_to_edge = half_width - abs(float(value) - center)
    return 1.0 + distance_to_edge / max(half_width, 1e-12)


def short_metric_name(metric: str) -> str:
    mapping = {
        "G1-V1 wind_speed_acf_max_abs_delta_lag1_12h": "Wind-speed autocorrelation",
        "G1-V2 p_event_fraction_gt_0p75": "Event-heavy windows",
        "G1-V2 p_event_fraction_lt_0p25": "Calm windows",
        "G1-V3a max_ks_antaws_scalar_base_variables": "Antarctic AWS distribution KS",
        "G1-V3b event_duration_median_h": "Event duration",
        "G1-V3b flux_wind_loglog_slope": "Flux-wind relationship",
        "G1-V3b diameter_wind_spearman_rho": "Particle-size-wind correlation",
        "G1-V4 wind_speed_psd_log_mse_0p1_4cpd": "Wind-speed power spectrum",
    }
    return mapping.get(str(metric), str(metric).replace("_", " "))


def format_value(value: float, rule: str) -> str:
    if abs(value) >= 10:
        val = f"{value:.1f}"
    elif abs(value) >= 1:
        val = f"{value:.2f}"
    else:
        val = f"{value:.3f}"
    return f"{val} ({rule})"


def plot_marginal(ax: plt.Axes, anchor: pd.DataFrame, synthetic: pd.DataFrame, column: str, title: str, unit: str) -> None:
    real = anchor[column].to_numpy(dtype=float)
    synth = synthetic[column].to_numpy(dtype=float)
    real = real[np.isfinite(real)]
    synth = synth[np.isfinite(synth)]
    lo = float(np.nanpercentile(np.concatenate([real, synth]), 0.5))
    hi = float(np.nanpercentile(np.concatenate([real, synth]), 99.5))
    bins = np.linspace(lo, hi, 42)
    ax.hist(real, bins=bins, density=True, histtype="step", linewidth=1.6, color=PAPER["ink"], label="Antarctic AWS reference")
    ax.hist(
        synth,
        bins=bins,
        density=True,
        alpha=0.86,
        color=PAPER["blue_fill"],
        edgecolor=PAPER["blue"],
        linewidth=0.45,
        label="Simulated",
    )
    ax.set_title(title, loc="left")
    ax.set_xlabel(unit)
    ax.set_ylabel("density")
    soften_axes(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw the paper simulation-validation figure.")
    parser.add_argument("--assets-dir", default="rl_sensor_scheduling_framework/reports/v3_supplement_assets")
    parser.add_argument("--antaws-root", default="data/AntAWS/3_hourly")
    parser.add_argument("--out-dir", default="rl_sensor_scheduling_framework/paper/figures")
    parser.add_argument("--window", type=int, default=512)
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic = pd.read_csv(assets_dir / "g1_v31_synthetic_truth.csv")
    validation = pd.read_csv(assets_dir / "exp_g1_generator_validation.csv")
    metadata = json.loads((assets_dir / "g1_v31_synthetic_metadata.json").read_text(encoding="utf-8"))
    stations = tuple(str(s) for s in metadata.get("stations", ["Panda100", "Panda200", "Taishan"]))
    freq_s = int(metadata.get("freq_s", 3600))
    anchor_full = build_antaws_anchor(Path(args.antaws_root), stations, freq_s=freq_s)
    n = min(len(anchor_full), len(synthetic))
    anchor_seq = anchor_full.iloc[:n].reset_index(drop=True)
    synthetic_seq = synthetic.iloc[:n].reset_index(drop=True)

    apply_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(5.35, 3.75), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.045, h_pad=0.045, hspace=0.08, wspace=0.08)

    plot_marginal(axes[0, 0], anchor_full, synthetic, "air_temperature_c", "(a) Air temperature", "temperature (deg C)")
    plot_marginal(axes[0, 1], anchor_full, synthetic, "wind_speed_ms", "(b) Wind speed", "wind speed (m/s)")
    axes[0, 0].legend(frameon=False, loc="upper left", handlelength=1.4)

    max_lag = 12
    lags = np.arange(1, max_lag + 1)
    anchor_acf = acf(anchor_seq["wind_speed_ms"].to_numpy(dtype=float), max_lag=max_lag)
    synth_acf = acf(synthetic_seq["wind_speed_ms"].to_numpy(dtype=float), max_lag=max_lag)
    axes[1, 0].plot(lags, anchor_acf, color=PAPER["ink"], linewidth=1.55, label="Antarctic AWS reference")
    axes[1, 0].plot(lags, synth_acf, color=PAPER["green"], linewidth=1.65, label="Simulated")
    axes[1, 0].fill_between(
        lags,
        anchor_acf - 0.05,
        anchor_acf + 0.05,
        color=PAPER["green_fill"],
        alpha=0.92,
        label="ACF tolerance",
    )
    axes[1, 0].set_title("(c) Wind autocorrelation", loc="left")
    axes[1, 0].set_xlabel("lag (h)")
    axes[1, 0].set_ylabel("ACF")
    axes[1, 0].set_ylim(0.0, 1.05)
    soften_axes(axes[1, 0])
    axes[1, 0].legend(frameon=False, loc="lower left")

    fractions = window_event_fractions(synthetic["event_flag"].to_numpy(dtype=bool), int(args.window))
    bins = np.linspace(0.0, 1.0, 31)
    weights = np.ones_like(fractions, dtype=float) / max(float(fractions.size), 1.0)
    axes[1, 1].axvspan(0.0, 0.25, color=PAPER["green_fill"], alpha=0.68, label="calm")
    axes[1, 1].axvspan(0.75, 1.0, color=PAPER["red_fill"], alpha=0.72, label="event-heavy")
    axes[1, 1].hist(
        fractions,
        bins=bins,
        weights=weights,
        color=PAPER["blue"],
        edgecolor=PAPER["blue"],
        linewidth=0.3,
        alpha=0.78,
    )
    axes[1, 1].set_title("(d) Event fractions in 512-step windows", loc="left")
    axes[1, 1].set_xlabel("event fraction")
    axes[1, 1].set_ylabel("fraction of windows")
    soften_axes(axes[1, 1])
    axes[1, 1].legend(frameon=False, loc="upper center", ncol=2)

    pdf_path = out_dir / "figure3_synthetic_statistics.pdf"
    png_path = out_dir / "figure3_synthetic_statistics.png"
    svg_path = out_dir / "figure3_synthetic_statistics.svg"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png_path, dpi=360, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(pdf_path)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
