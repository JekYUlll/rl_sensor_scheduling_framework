#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_sources.public_weather_synthesis import (  # noqa: E402
    PublicWeatherSynthesisConfig,
    build_antaws_anchor,
    generate_public_weather_truth,
    validate_synthetic_against_anchor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an AntAWS-anchored windblown truth CSV.")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--freq-s", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-keep-fraction", type=float, default=0.15)
    parser.add_argument("--blowing-snow-event-coverage", type=float, default=0.30)
    parser.add_argument("--blowing-snow-event-model", default="clustered")
    parser.add_argument("--blowing-snow-min-duration-steps", type=int, default=10)
    parser.add_argument("--blowing-snow-max-duration-steps", type=int, default=30)
    parser.add_argument("--blowing-snow-min-gap-steps", type=int, default=6)
    parser.add_argument("--blowing-snow-lead-steps", type=int, default=5)
    parser.add_argument("--blowing-snow-wind-margin-ms", type=float, default=1.5)
    parser.add_argument("--cred-hysteresis-on", type=float, default=0.6)
    parser.add_argument("--cred-hysteresis-off", type=float, default=0.3)
    parser.add_argument("--flux-wind-exponent", type=float, default=3.6)
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
    parser.add_argument("--out", default="data/generated/public_weather_truth.csv")
    parser.add_argument("--report-dir", default="reports/datasets/public_weather_truth")
    args = parser.parse_args()

    cfg = PublicWeatherSynthesisConfig(
        antaws_root=Path(args.antaws_root),
        stations=tuple(str(s) for s in args.stations),
        steps=int(args.steps),
        freq_s=int(args.freq_s),
        seed=int(args.seed),
        phase_keep_fraction=float(args.phase_keep_fraction),
        blowing_snow_event_coverage=float(args.blowing_snow_event_coverage),
        blowing_snow_event_model=str(args.blowing_snow_event_model),
        blowing_snow_min_duration_steps=int(args.blowing_snow_min_duration_steps),
        blowing_snow_max_duration_steps=int(args.blowing_snow_max_duration_steps),
        blowing_snow_min_gap_steps=int(args.blowing_snow_min_gap_steps),
        blowing_snow_lead_steps=int(args.blowing_snow_lead_steps),
        blowing_snow_wind_margin_ms=float(args.blowing_snow_wind_margin_ms),
        cred_hysteresis_on=float(args.cred_hysteresis_on),
        cred_hysteresis_off=float(args.cred_hysteresis_off),
        flux_wind_exponent=float(args.flux_wind_exponent),
        event_microstructure_sigma=float(args.event_microstructure_sigma),
        event_microstructure_alpha=float(args.event_microstructure_alpha),
        event_microstructure_diameter_scale=float(args.event_microstructure_diameter_scale),
        event_microstructure_velocity_scale=float(args.event_microstructure_velocity_scale),
        event_particle_microstructure_correlation=float(args.event_particle_microstructure_correlation),
        event_subtypes_enabled=bool(args.event_subtypes_enabled),
        event_subtype_assignment=str(args.event_subtype_assignment),
        event_subtype_particle_min_parsivel_availability=float(
            args.event_subtype_particle_min_parsivel_availability
        ),
        event_subtype_particle_prob=float(args.event_subtype_particle_prob),
        event_subtype_flux_prob=float(args.event_subtype_flux_prob),
        event_subtype_thermal_prob=float(args.event_subtype_thermal_prob),
        event_subtype_particle_flux_multiplier=float(args.event_subtype_particle_flux_multiplier),
        event_subtype_flux_multiplier=float(args.event_subtype_flux_multiplier),
        event_subtype_thermal_flux_multiplier=float(args.event_subtype_thermal_flux_multiplier),
        event_subtype_particle_diameter_shift_mm=float(args.event_subtype_particle_diameter_shift_mm),
        event_subtype_particle_velocity_boost_ms=float(args.event_subtype_particle_velocity_boost_ms),
        event_subtype_flux_diameter_shift_mm=float(args.event_subtype_flux_diameter_shift_mm),
        event_subtype_flux_velocity_boost_ms=float(args.event_subtype_flux_velocity_boost_ms),
        event_subtype_thermal_surface_drop_c=float(args.event_subtype_thermal_surface_drop_c),
        event_subtype_particle_humidity_boost_pct=float(args.event_subtype_particle_humidity_boost_pct),
        event_subtype_flux_wind_boost_ms=float(args.event_subtype_flux_wind_boost_ms),
        event_subtype_thermal_air_temp_drop_c=float(args.event_subtype_thermal_air_temp_drop_c),
        event_subtype_latent_alpha=float(args.event_subtype_latent_alpha),
        event_subtype_particle_latent_diameter_scale_mm=float(
            args.event_subtype_particle_latent_diameter_scale_mm
        ),
        event_subtype_particle_latent_velocity_scale_ms=float(
            args.event_subtype_particle_latent_velocity_scale_ms
        ),
        event_subtype_flux_latent_sigma=float(args.event_subtype_flux_latent_sigma),
        event_subtype_flux_latent_linear_scale=float(args.event_subtype_flux_latent_linear_scale),
        event_subtype_flux_latent_linear_offset=float(args.event_subtype_flux_latent_linear_offset),
        event_subtype_flux_latent_linear_clip=float(args.event_subtype_flux_latent_linear_clip),
        event_subtype_thermal_latent_surface_scale_c=float(
            args.event_subtype_thermal_latent_surface_scale_c
        ),
        event_subtype_latent_target_lag_steps=int(args.event_subtype_latent_target_lag_steps),
        event_subtype_context_lead_steps=int(args.event_subtype_context_lead_steps),
        event_subtype_context_noise_std=float(args.event_subtype_context_noise_std),
        event_subtype_context_latent_strength=float(args.event_subtype_context_latent_strength),
    )
    df, meta = generate_public_weather_truth(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    anchor = build_antaws_anchor(cfg.antaws_root, cfg.stations, freq_s=cfg.freq_s)
    validation = validate_synthetic_against_anchor(anchor, df)
    validation.to_csv(report_dir / "synthetic_validation.csv", index=False)
    (report_dir / "synthetic_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(out_path)
    print(report_dir / "synthetic_validation.csv")


if __name__ == "__main__":
    main()
