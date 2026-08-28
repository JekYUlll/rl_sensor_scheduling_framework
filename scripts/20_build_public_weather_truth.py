#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_sources.public_weather_synthesis import (  # noqa: E402
    PublicWeatherSynthesisConfig,
    build_antaws_anchor,
    generate_public_weather_truth,
    validate_synthetic_against_anchor,
)

DEFAULT_QUALITY_SENSOR_IDS = (
    "met_station_core", "radiometer_basic", "shielded_thermo_hygro",
    "surface_temp_ir", "laser_disdrometer", "fc4_flux",
)


def add_channel_quality_dynamics(
    frame,
    *,
    sensor_ids: tuple[str, ...],
    seed: int,
    coverage: float,
    min_duration: int,
    max_duration: int,
    min_gap: int,
    degraded_quality: float,
    transition_steps: int,
    report_noise_std: float,
    mode: str = "independent",
):
    """Add bounded online channel-quality signals to the generated truth frame.

    ``independent`` retains the original slow, sensor-specific outage process.
    ``condition_dependent`` models exposure-dependent degradation from continuous
    weather and transport drivers. ``condition_dependent_crossover`` is the
    physical-group variant: each installed instrument has a different
    weather-dependent reliability mechanism, so no group is assumed to be
    uniformly most useful. Neither mode changes the event process; both produce
    noisy diagnostics that determine measurement reliability.
    """
    out = frame.copy()
    steps = len(out)
    low = float(np.clip(degraded_quality, 0.0, 1.0))
    if mode not in {"independent", "condition_dependent", "condition_dependent_crossover"}:
        raise ValueError(
            "channel quality mode must be independent, condition_dependent, or "
            "condition_dependent_crossover"
        )
    if mode in {"condition_dependent", "condition_dependent_crossover"}:
        required = {
            "wind_speed_ms",
            "relative_humidity",
            "event_subtype_particle_latent",
            "event_subtype_flux_latent",
            "event_subtype_thermal_latent",
        }
        if mode == "condition_dependent_crossover":
            required.update({"air_temperature_c", "solar_radiation_wm2"})
        missing = sorted(required.difference(out.columns))
        if missing:
            raise ValueError(f"condition-dependent quality requires columns: {missing}")

        def positive_unit(column: str) -> np.ndarray:
            values = np.maximum(out[column].to_numpy(dtype=float), 0.0)
            scale = max(float(np.quantile(values, 0.95)), 1.0e-6)
            return np.clip(values / scale, 0.0, 1.0)

        wind = positive_unit("wind_speed_ms")
        humidity = np.clip(
            (out["relative_humidity"].to_numpy(dtype=float) - 60.0) / 30.0,
            0.0,
            1.0,
        )
        if mode == "condition_dependent_crossover":
            temperature = out["air_temperature_c"].to_numpy(dtype=float)
            radiation = positive_unit("solar_radiation_wm2")
            cold = np.clip((-temperature - 5.0) / 20.0, 0.0, 1.0)
            icing = humidity * cold
            severe_wind = np.clip((wind - 0.55) / 0.45, 0.0, 1.0)
            moderate_wind = np.clip(1.0 - np.abs(wind - 0.52) / 0.35, 0.0, 1.0)
            low_transport_signal = 1.0 - np.clip(
                0.25 * wind + 0.75 * severe_wind, 0.0, 1.0
            )
            # Fixed physical reliability assumptions. GMX and SI-111 are
            # susceptible to icing or wet exposure; the pyranometer has lower
            # useful signal under diffuse humid conditions; Parsivel is most
            # reliable in moderate transport and suffers optical deposition in
            # severe wind; FlowCapt's signal-to-noise improves with transport.
            # All drivers are ordinary meteorological variables, not labels.
            exposure_profiles = {
                "gmx500_weather_station": 0.70 * icing + 0.15 * severe_wind,
                "lps10_pyranometer": 0.60 * (1.0 - radiation) + 0.25 * humidity,
                "si111_surface_ir": 0.65 * icing + 0.15 * severe_wind,
                "parsivel2_disdrometer": (
                    0.65 * severe_wind + 0.20 * humidity + 0.10 * (1.0 - moderate_wind)
                ),
                "flowcapt_fc4": 0.75 * low_transport_signal + 0.10 * icing,
            }
        else:
            particle = positive_unit("event_subtype_particle_latent")
            flux = positive_unit("event_subtype_flux_latent")
            thermal = positive_unit("event_subtype_thermal_latent")
            # Each profile represents environmental exposure of an installed
            # instrument, not a subtype-to-sensor assignment. The historical
            # logical IDs remain for backwards-compatible archive regeneration.
            exposure_profiles = {
                "met_station_core": 0.12 * wind + 0.08 * humidity,
                "radiometer_basic": 0.14 * humidity + 0.10 * thermal,
                "shielded_thermo_hygro": 0.10 * wind + 0.16 * humidity,
                "surface_temp_ir": 0.55 * particle + 0.20 * flux + 0.08 * humidity,
                "laser_disdrometer": 0.12 * particle + 0.60 * flux + 0.08 * wind,
                "fc4_flux": 0.58 * particle + 0.12 * flux + 0.10 * wind,
                "gmx500_weather_station": 0.10 * wind + 0.15 * humidity,
                "lps10_pyranometer": 0.25 * wind + 0.35 * humidity,
                "si111_surface_ir": 0.15 * wind + 0.40 * humidity,
                "parsivel2_disdrometer": 0.65 * wind + 0.20 * humidity,
                "flowcapt_fc4": 0.15 * wind + 0.12 * humidity,
            }
        for sensor_idx, sensor_id in enumerate(sensor_ids):
            rng = np.random.default_rng(int(seed) + 70001 + 1009 * sensor_idx)
            exposure = exposure_profiles.get(str(sensor_id), 0.15 * wind + 0.10 * humidity)
            quality = 1.0 - (1.0 - low) * np.clip(exposure, 0.0, 1.0)
            if float(report_noise_std) > 0.0:
                quality = quality + rng.normal(0.0, float(report_noise_std), size=steps)
            out[f"agent_context_quality_{sensor_id}"] = np.clip(quality, low, 1.0)
        return out

    target = int(round(float(np.clip(coverage, 0.0, 0.95)) * steps))
    min_len = max(1, int(min_duration))
    max_len = max(min_len, int(max_duration))
    gap = max(0, int(min_gap))
    transition = max(0, int(transition_steps))
    for sensor_idx, sensor_id in enumerate(sensor_ids):
        rng = np.random.default_rng(int(seed) + 70001 + 1009 * sensor_idx)
        quality = np.ones(steps, dtype=float)
        occupied = 0
        attempts = 0
        while occupied < target and attempts < max(100, steps * 8):
            attempts += 1
            duration = int(rng.integers(min_len, max_len + 1))
            start = int(rng.integers(0, max(1, steps - duration + 1)))
            end = min(steps, start + duration)
            left = max(0, start - gap)
            right = min(steps, end + gap)
            if np.any(quality[left:right] < 1.0):
                continue
            quality[start:end] = low
            ramp = min(transition, max(0, (end - start) // 2))
            if ramp > 0:
                down = np.linspace(1.0, low, ramp + 2, dtype=float)[1:-1]
                up = np.linspace(low, 1.0, ramp + 2, dtype=float)[1:-1]
                quality[start : start + ramp] = down
                quality[end - ramp : end] = up
            occupied += end - start
        if float(report_noise_std) > 0.0:
            quality = quality + rng.normal(0.0, float(report_noise_std), size=steps)
        out[f"agent_context_quality_{sensor_id}"] = np.clip(quality, 0.0, 1.0)
    return out


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
    parser.add_argument(
        "--event-subtype-assignment",
        choices=["random", "stratified", "stratified_duration"],
        default="random",
    )
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
    parser.add_argument("--nowcast-lead-steps", type=int, default=0)
    parser.add_argument("--nowcast-wind-noise-std", type=float, default=1.0)
    parser.add_argument("--nowcast-humidity-noise-std", type=float, default=3.0)
    parser.add_argument("--nowcast-temperature-noise-std", type=float, default=0.7)
    parser.add_argument("--channel-quality-enabled", action="store_true")
    parser.add_argument(
        "--channel-quality-mode",
        choices=["independent", "condition_dependent", "condition_dependent_crossover"],
        default="independent",
    )
    parser.add_argument("--channel-quality-sensor-ids", nargs="+", default=list(DEFAULT_QUALITY_SENSOR_IDS))
    parser.add_argument("--channel-quality-degraded-coverage", type=float, default=0.0)
    parser.add_argument("--channel-quality-min-duration-steps", type=int, default=12)
    parser.add_argument("--channel-quality-max-duration-steps", type=int, default=48)
    parser.add_argument("--channel-quality-min-gap-steps", type=int, default=12)
    parser.add_argument("--channel-quality-degraded-value", type=float, default=0.2)
    parser.add_argument("--channel-quality-transition-steps", type=int, default=0)
    parser.add_argument("--channel-quality-report-noise-std", type=float, default=0.02)
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
        nowcast_lead_steps=int(args.nowcast_lead_steps),
        nowcast_wind_noise_std=float(args.nowcast_wind_noise_std),
        nowcast_humidity_noise_std=float(args.nowcast_humidity_noise_std),
        nowcast_temperature_noise_std=float(args.nowcast_temperature_noise_std),
    )
    df, meta = generate_public_weather_truth(cfg)
    if bool(args.channel_quality_enabled):
        df = add_channel_quality_dynamics(
            df,
            sensor_ids=tuple(str(value) for value in args.channel_quality_sensor_ids),
            seed=int(args.seed),
            coverage=float(args.channel_quality_degraded_coverage),
            min_duration=int(args.channel_quality_min_duration_steps),
            max_duration=int(args.channel_quality_max_duration_steps),
            min_gap=int(args.channel_quality_min_gap_steps),
            degraded_quality=float(args.channel_quality_degraded_value),
            transition_steps=int(args.channel_quality_transition_steps),
            report_noise_std=float(args.channel_quality_report_noise_std),
            mode=str(args.channel_quality_mode),
        )
        meta["channel_quality"] = {
            "enabled": True,
            "mode": str(args.channel_quality_mode),
            "sensor_ids": [str(value) for value in args.channel_quality_sensor_ids],
            "degraded_coverage": float(args.channel_quality_degraded_coverage),
            "min_duration_steps": int(args.channel_quality_min_duration_steps),
            "max_duration_steps": int(args.channel_quality_max_duration_steps),
            "min_gap_steps": int(args.channel_quality_min_gap_steps),
            "degraded_value": float(args.channel_quality_degraded_value),
            "transition_steps": int(args.channel_quality_transition_steps),
            "report_noise_std": float(args.channel_quality_report_noise_std),
        }

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
