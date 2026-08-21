#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_sources.public_weather_synthesis import (  # noqa: E402
    PublicWeatherSynthesisConfig,
    _acf,
    _bool_runs,
    build_antaws_anchor,
    generate_public_weather_truth,
)

_G1_SPEC = importlib.util.spec_from_file_location("g1_validation", ROOT / "scripts" / "39_v2_validate_v31_generator.py")
if _G1_SPEC is None or _G1_SPEC.loader is None:
    raise RuntimeError("Could not load scripts/39_v2_validate_v31_generator.py")
_G1_MODULE = importlib.util.module_from_spec(_G1_SPEC)
_G1_SPEC.loader.exec_module(_G1_MODULE)

BASE_VARIABLES = _G1_MODULE.BASE_VARIABLES
_linear_slope_loglog = _G1_MODULE._linear_slope_loglog
_log_psd_mse = _G1_MODULE._log_psd_mse
_window_event_fractions = _G1_MODULE._window_event_fractions


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


def _make_cfg(args: argparse.Namespace, *, model: str) -> PublicWeatherSynthesisConfig:
    return PublicWeatherSynthesisConfig(
        antaws_root=Path(args.antaws_root),
        stations=tuple(str(s) for s in args.stations),
        steps=int(args.steps),
        freq_s=int(args.freq_s),
        seed=int(args.seed),
        phase_keep_fraction=0.15,
        blowing_snow_event_coverage=float(args.event_coverage),
        blowing_snow_event_model=model,
        blowing_snow_min_duration_steps=int(args.min_duration),
        blowing_snow_max_duration_steps=int(args.max_duration),
        blowing_snow_min_gap_steps=int(args.min_gap),
        blowing_snow_lead_steps=int(args.lead_steps),
        blowing_snow_wind_margin_ms=float(args.wind_margin_ms),
        flux_wind_exponent=float(args.flux_wind_exponent),
    )


def _metrics(label: str, cfg: PublicWeatherSynthesisConfig, anchor: pd.DataFrame, window: int) -> dict[str, object]:
    synthetic, meta = generate_public_weather_truth(cfg)
    n = min(len(anchor), len(synthetic))
    anchor_wind = anchor["wind_speed_ms"].to_numpy(dtype=float)[:n]
    synthetic_wind = synthetic["wind_speed_ms"].to_numpy(dtype=float)[:n]
    max_lag = max(1, int(round(12 * 3600 / float(cfg.freq_s))))
    event_flag = synthetic["event_flag"].to_numpy(dtype=bool)
    event_fractions = _window_event_fractions(event_flag, int(window))
    heavy_successes = int(np.sum(event_fractions > 0.75))
    calm_successes = int(np.sum(event_fractions < 0.25))
    event_runs = _bool_runs(event_flag)
    durations_h = np.asarray([(end - start) * int(cfg.freq_s) / 3600.0 for start, end in event_runs])
    wind = synthetic["wind_speed_ms"].to_numpy(dtype=float)
    flux = synthetic["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float)
    diameter = synthetic["snow_particle_mean_diameter_mm"].to_numpy(dtype=float)
    active = event_flag
    event_flux_mask = active & (flux > 0.0) & (wind > 0.0)
    particle_mask = active & (diameter > 0.0)
    ks_values = [
        float(stats.ks_2samp(anchor[col].to_numpy(dtype=float), synthetic[col].to_numpy(dtype=float)).statistic)
        for col in BASE_VARIABLES
    ]
    heavy_ci_low, heavy_ci_high = _wilson_interval(heavy_successes, int(event_fractions.size))
    return {
        "generator": label,
        "event_model": str(meta.get("blowing_snow_event_model", cfg.blowing_snow_event_model)),
        "steps": int(cfg.steps),
        "window": int(window),
        "n_windows": int(event_fractions.size),
        "wind_speed_acf_max_abs_delta_lag1_12h": float(
            np.max(np.abs(_acf(anchor_wind, max_lag=max_lag) - _acf(synthetic_wind, max_lag=max_lag)))
        ),
        "p_event_fraction_gt_0p75": float(np.mean(event_fractions > 0.75)),
        "p_event_fraction_gt_0p75_ci95_low": heavy_ci_low,
        "p_event_fraction_gt_0p75_ci95_high": heavy_ci_high,
        "p_event_fraction_lt_0p25": float(np.mean(event_fractions < 0.25)),
        "max_event_fraction_512": float(np.max(event_fractions)),
        "event_fraction_mean": float(np.mean(event_fractions)),
        "event_cluster_count": int(len(event_runs)),
        "event_duration_median_h": float(np.median(durations_h)) if durations_h.size else float("nan"),
        "max_ks_antaws_scalar_base_variables": float(np.max(ks_values)),
        "flux_wind_loglog_slope": _linear_slope_loglog(wind[event_flux_mask], flux[event_flux_mask]),
        "diameter_wind_spearman_rho": float(stats.spearmanr(wind[particle_mask], diameter[particle_mask]).correlation)
        if int(np.sum(particle_mask)) >= 4
        else float("nan"),
        "wind_speed_psd_log_mse_0p1_4cpd": _log_psd_mse(anchor_wind, synthetic_wind, freq_s=int(cfg.freq_s)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare V2 and V3.1 generator diagnostics.")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--freq-s", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--event-coverage", type=float, default=0.28)
    parser.add_argument("--min-duration", type=int, default=12)
    parser.add_argument("--max-duration", type=int, default=24)
    parser.add_argument("--min-gap", type=int, default=4)
    parser.add_argument("--lead-steps", type=int, default=6)
    parser.add_argument("--wind-margin-ms", type=float, default=1.2)
    parser.add_argument("--flux-wind-exponent", type=float, default=3.0)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--out-dir", default="reports/v3_supplement_assets")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = build_antaws_anchor(Path(args.antaws_root), tuple(str(s) for s in args.stations), freq_s=int(args.freq_s))
    rows = [
        _metrics("V2 clustered diagnostic", _make_cfg(args, model="clustered"), anchor, int(args.window)),
        _metrics("V3.1 semi-Markov diagnostic", _make_cfg(args, model="semi_markov"), anchor, int(args.window)),
    ]
    result = pd.DataFrame(rows)
    path = out_dir / "exp_g1_v2_v31_generator_comparison.csv"
    result.to_csv(path, index=False)
    print(result.to_string(index=False))
    print(path)


if __name__ == "__main__":
    main()
