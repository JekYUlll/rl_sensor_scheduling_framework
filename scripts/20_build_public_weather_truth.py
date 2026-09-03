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
    forecast_quality: bool = False,
):
    """Add bounded online channel-quality signals to the generated truth frame.

    ``independent`` retains the original slow, sensor-specific outage process.
    ``condition_dependent`` models exposure-dependent degradation from continuous
    weather and transport drivers. ``condition_dependent_crossover`` and
    ``condition_dependent_crossover_strong`` and
    ``condition_dependent_crossover_balanced``,
    ``condition_dependent_crossover_calibrated``, and
    ``condition_dependent_crossover_robust`` are physical-group variants:
    physical-group variant: each installed instrument has a different
    weather-dependent reliability mechanism, so no group is assumed to be
    uniformly most useful. Neither mode changes the event process; both produce
    noisy diagnostics that determine measurement reliability.
    """
    out = frame.copy()
    steps = len(out)
    low = float(np.clip(degraded_quality, 0.0, 1.0))
    if mode not in {
        "independent", "condition_dependent", "condition_dependent_crossover",
        "condition_dependent_crossover_strong",
        "condition_dependent_crossover_balanced",
        "condition_dependent_crossover_calibrated",
        "condition_dependent_crossover_robust",
    }:
        raise ValueError(
            "channel quality mode must be independent, condition_dependent, "
            "condition_dependent_crossover, condition_dependent_crossover_strong, "
            "or condition_dependent_crossover_balanced"
            " or condition_dependent_crossover_calibrated"
            " or condition_dependent_crossover_robust"
        )
    if mode in {
        "condition_dependent", "condition_dependent_crossover",
        "condition_dependent_crossover_strong",
        "condition_dependent_crossover_balanced",
        "condition_dependent_crossover_calibrated",
        "condition_dependent_crossover_robust",
    }:
        required = {
            "wind_speed_ms",
            "relative_humidity",
        }
        if mode == "condition_dependent":
            required.update({
                "event_subtype_particle_latent",
                "event_subtype_flux_latent",
                "event_subtype_thermal_latent",
            })
        if mode in {
            "condition_dependent_crossover", "condition_dependent_crossover_strong",
            "condition_dependent_crossover_balanced",
            "condition_dependent_crossover_calibrated",
            "condition_dependent_crossover_robust",
        }:
            required.update({"air_temperature_c", "solar_radiation_wm2"})
        missing = sorted(required.difference(out.columns))
        if missing:
            raise ValueError(f"condition-dependent quality requires columns: {missing}")

        def positive_unit(column: str, *, scale: float | None = None) -> np.ndarray:
            values = np.maximum(out[column].to_numpy(dtype=float), 0.0)
            effective_scale = (
                float(scale)
                if scale is not None
                else max(float(np.quantile(values, 0.95)), 1.0e-6)
            )
            return np.clip(values / max(effective_scale, 1.0e-6), 0.0, 1.0)

        def profiles_for(
            prefix: str = "",
            *,
            normalization_scales: dict[str, float] | None = None,
        ) -> dict[str, np.ndarray]:
            wind_column = f"{prefix}wind_speed_ms" if prefix else "wind_speed_ms"
            humidity_column = f"{prefix}relative_humidity" if prefix else "relative_humidity"
            temperature_column = f"{prefix}air_temperature_c" if prefix else "air_temperature_c"
            radiation_column = (
                f"{prefix}solar_radiation_wm2" if prefix else "solar_radiation_wm2"
            )
            required_columns = {wind_column, humidity_column, temperature_column, radiation_column}
            missing_columns = sorted(required_columns.difference(out.columns))
            if missing_columns:
                raise ValueError(f"quality profiles require columns: {missing_columns}")
            scales = normalization_scales or {}
            wind = positive_unit(wind_column, scale=scales.get("wind"))
            humidity = np.clip(
                (out[humidity_column].to_numpy(dtype=float) - 60.0) / 30.0,
                0.0,
                1.0,
            )
            temperature = out[temperature_column].to_numpy(dtype=float)
            radiation = positive_unit(radiation_column, scale=scales.get("radiation"))
            cold = np.clip((-temperature - 5.0) / 20.0, 0.0, 1.0)
            icing = humidity * cold
            severe_wind = np.clip((wind - 0.55) / 0.45, 0.0, 1.0)
            moderate_wind = np.clip(1.0 - np.abs(wind - 0.52) / 0.35, 0.0, 1.0)
            low_transport_signal = 1.0 - np.clip(
                0.25 * wind + 0.75 * severe_wind, 0.0, 1.0
            )
            if mode == "condition_dependent_crossover":
                profiles = {
                    "met_station_core": 0.70 * icing + 0.15 * severe_wind,
                    "radiometer_basic": 0.60 * (1.0 - radiation) + 0.25 * humidity,
                    "shielded_thermo_hygro": 0.35 * icing + 0.25 * humidity + 0.10 * severe_wind,
                    "surface_temp_ir": 0.65 * icing + 0.15 * severe_wind,
                    "laser_disdrometer": (
                        0.65 * severe_wind + 0.20 * humidity + 0.10 * (1.0 - moderate_wind)
                    ),
                    "fc4_flux": 0.75 * low_transport_signal + 0.10 * icing,
                }
                profiles.update({
                    "gmx500_weather_station": profiles["met_station_core"],
                    "lps10_pyranometer": profiles["radiometer_basic"],
                    "si111_surface_ir": profiles["surface_temp_ir"],
                    "parsivel2_disdrometer": profiles["laser_disdrometer"],
                    "flowcapt_fc4": profiles["fc4_flux"],
                })
                return profiles
            exposure_profiles = {
                "met_station_core": 0.80 * severe_wind + 0.20 * humidity,
                "radiometer_basic": 0.80 * (1.0 - radiation) + 0.20 * humidity,
                "shielded_thermo_hygro": 0.35 * icing + 0.25 * humidity + 0.10 * severe_wind,
                "surface_temp_ir": 0.80 * icing + 0.20 * severe_wind,
                "laser_disdrometer": 0.80 * (1.0 - moderate_wind) + 0.20 * humidity,
                "fc4_flux": 0.80 * low_transport_signal + 0.20 * icing,
            }
            if mode == "condition_dependent_crossover_calibrated":
                # Fixed instrument-response calibration.  The offsets and
                # scales are scene-independent engineering assumptions that
                # map each physical exposure proxy onto a comparable dynamic
                # operating range; they are not fitted to event labels or
                # evaluation outcomes.
                calibration = {
                    "met_station_core": (0.25, 0.45),
                    "radiometer_basic": (0.57, 0.45),
                    "shielded_thermo_hygro": (0.18, 0.25),
                    "surface_temp_ir": (0.20, 0.30),
                    "laser_disdrometer": (0.35, 0.45),
                    "fc4_flux": (0.45, 0.45),
                }
                exposure_profiles = {
                    sensor: np.clip(
                        (profile - calibration[sensor][0])
                        / calibration[sensor][1],
                        0.0,
                        1.0,
                    )
                    for sensor, profile in exposure_profiles.items()
                }
            if mode == "condition_dependent_crossover_robust":
                # Fixed, scene-independent response calibration for the
                # installed instruments. Profiles remain functions of
                # observable weather drivers; offsets and scales are not
                # fitted to event labels or evaluation outcomes.
                transport = np.clip(
                    0.25 * wind + 0.75 * severe_wind,
                    0.0,
                    1.0,
                )
                exposure_profiles.update(
                    {
                        "surface_temp_ir": 0.80 * (1.0 - icing) + 0.20 * severe_wind,
                        "fc4_flux": 0.50 * (1.0 - transport)
                        + 0.20 * icing
                        + 0.10 * (1.0 - moderate_wind),
                    }
                )
                calibration = {
                    "met_station_core": (0.05, 0.50),
                    "radiometer_basic": (0.50, 0.50),
                    "shielded_thermo_hygro": (0.15, 0.50),
                    "surface_temp_ir": (0.50, 0.50),
                    "laser_disdrometer": (0.22, 0.50),
                    "fc4_flux": (0.30, 0.50),
                }
                exposure_profiles = {
                    sensor: np.clip(
                        (profile - calibration[sensor][0])
                        / calibration[sensor][1],
                        0.0,
                        1.0,
                    )
                    for sensor, profile in exposure_profiles.items()
                }
            if mode == "condition_dependent_crossover_balanced":
                profile_ids = list(exposure_profiles)
                raw = np.vstack([exposure_profiles[sensor] for sensor in profile_ids])
                centered = raw - np.mean(raw, axis=0, keepdims=True)
                balanced = np.clip(0.45 + 0.90 * centered, 0.05, 0.95)
                exposure_profiles = {
                    sensor: balanced[idx] for idx, sensor in enumerate(profile_ids)
                }
            exposure_profiles.update({
                "gmx500_weather_station": exposure_profiles["met_station_core"],
                "lps10_pyranometer": exposure_profiles["radiometer_basic"],
                "si111_surface_ir": exposure_profiles["surface_temp_ir"],
                "parsivel2_disdrometer": exposure_profiles["laser_disdrometer"],
                "flowcapt_fc4": exposure_profiles["fc4_flux"],
            })
            return exposure_profiles

        wind = positive_unit("wind_speed_ms")
        humidity = np.clip(
            (out["relative_humidity"].to_numpy(dtype=float) - 60.0) / 30.0,
            0.0,
            1.0,
        )
        if mode in {
            "condition_dependent_crossover",
            "condition_dependent_crossover_strong",
            "condition_dependent_crossover_balanced",
            "condition_dependent_crossover_calibrated",
            "condition_dependent_crossover_robust",
        }:
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
            # Each physical group has a distinct, weather-observable exposure
            # mechanism. The balanced variant centers these same profiles
            # across groups; it must be used for both actual and forecast
            # quality so the two columns describe one physical process.
            exposure_profiles = profiles_for()
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
        if forecast_quality:
            forecast_profiles = profiles_for(
                "agent_context_nowcast_",
                normalization_scales={
                    "wind": max(float(np.quantile(out["wind_speed_ms"].to_numpy(dtype=float), 0.95)), 1.0e-6),
                    "radiation": max(float(np.quantile(out["solar_radiation_wm2"].to_numpy(dtype=float), 0.95)), 1.0e-6),
                },
            )
            for sensor_idx, sensor_id in enumerate(sensor_ids):
                rng = np.random.default_rng(int(seed) + 91001 + 1009 * sensor_idx)
                exposure = forecast_profiles.get(
                    str(sensor_id),
                    0.15 * positive_unit("agent_context_nowcast_wind_speed_ms")
                    + 0.10 * np.clip(
                        (out["agent_context_nowcast_relative_humidity"].to_numpy(dtype=float) - 60.0)
                        / 30.0,
                        0.0,
                        1.0,
                    ),
                )
                predicted = 1.0 - (1.0 - low) * np.clip(exposure, 0.0, 1.0)
                if float(report_noise_std) > 0.0:
                    predicted = predicted + rng.normal(
                        0.0, float(report_noise_std), size=steps
                    )
                out[f"agent_context_quality_forecast_{sensor_id}"] = np.clip(
                    predicted, low, 1.0
                )
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


def add_nowcast_operating_state_dynamics(
    frame,
    *,
    sensor_ids: tuple[str, ...],
    transport_rho: float,
    thermal_rho: float,
    target_scale: float,
    quality_scale: float,
):
    """Apply persistent, nowcast-driven operating conditions to a truth frame.

    The two internal factors are deterministic recurrences of noisy nowcasts.
    They are retained only for generator audit; execution receives ordinary
    nowcasts and physical quality reports, never these latent columns.
    """
    out = frame.copy()
    required = (
        "agent_context_nowcast_wind_speed_ms",
        "agent_context_nowcast_relative_humidity",
        "agent_context_nowcast_air_temperature_c",
        "agent_context_nowcast_solar_radiation_wm2",
    )
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"continuous operating state requires noisy nowcasts: {missing}")

    def unit(values: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(values, [0.05, 0.95])
        return np.clip((values - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)

    wind = unit(out[required[0]].to_numpy(dtype=float))
    humidity = unit(out[required[1]].to_numpy(dtype=float))
    cold = unit(-out[required[2]].to_numpy(dtype=float))
    low_solar = 1.0 - unit(out[required[3]].to_numpy(dtype=float))
    transport_driver = np.clip(0.70 * wind + 0.30 * humidity, 0.0, 1.0)
    thermal_driver = np.clip(0.55 * cold + 0.30 * low_solar + 0.15 * humidity, 0.0, 1.0)

    def persistent(driver: np.ndarray, rho: float) -> np.ndarray:
        state = np.empty_like(driver, dtype=float)
        state[0] = float(driver[0])
        alpha = float(np.clip(rho, 0.0, 0.999))
        for idx in range(1, len(driver)):
            state[idx] = alpha * state[idx - 1] + (1.0 - alpha) * driver[idx]
        return np.clip(state, 0.0, 1.0)

    transport = persistent(transport_driver, transport_rho)
    thermal = persistent(thermal_driver, thermal_rho)
    transport_centered = 2.0 * transport - 1.0
    thermal_centered = 2.0 * thermal - 1.0
    target_gain = max(0.0, float(target_scale))
    out["snow_mass_flux_kg_m2_s"] = np.clip(
        out["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float)
        * np.exp(0.55 * target_gain * transport_centered),
        0.0,
        None,
    )
    out["snow_particle_mean_velocity_ms"] = np.clip(
        out["snow_particle_mean_velocity_ms"].to_numpy(dtype=float)
        + 1.20 * target_gain * transport_centered,
        0.0,
        20.0,
    )
    out["snow_particle_mean_diameter_mm"] = np.clip(
        out["snow_particle_mean_diameter_mm"].to_numpy(dtype=float)
        + 0.035 * target_gain * transport_centered,
        0.04,
        0.5,
    )
    out["snow_surface_temperature_c"] = (
        out["snow_surface_temperature_c"].to_numpy(dtype=float)
        + 1.10 * target_gain * thermal_centered
    )

    quality_gain = max(0.0, float(quality_scale))
    response = {
        "met_station_core": -0.16 * transport_centered - 0.08 * thermal_centered,
        "radiometer_basic": -0.06 * transport_centered - 0.16 * thermal_centered,
        "shielded_thermo_hygro": -0.04 * transport_centered + 0.10 * thermal_centered,
        "surface_temp_ir": -0.10 * transport_centered + 0.20 * thermal_centered,
        "laser_disdrometer": 0.16 * (1.0 - np.abs(transport_centered)) - 0.08 * thermal_centered,
        "fc4_flux": 0.25 * transport_centered - 0.06 * thermal_centered,
    }
    for sensor_id in sensor_ids:
        factor = np.clip(1.0 + quality_gain * response.get(str(sensor_id), 0.0), 0.55, 1.25)
        for prefix in ("agent_context_quality_", "agent_context_quality_forecast_"):
            column = f"{prefix}{sensor_id}"
            if column in out.columns:
                out[column] = np.clip(out[column].to_numpy(dtype=float) * factor, 0.05, 1.0)
    out["generator_operating_transport_state"] = transport
    out["generator_operating_thermal_state"] = thermal
    return out


def add_exposure_recovery_dynamics(
    frame,
    *,
    sensor_ids: tuple[str, ...],
    balanced_recovery: bool = False,
    decoupled_drivers: bool = False,
    target_gain: float = 1.0,
    low_frequency_targets: bool = False,
    residual_fraction: float = 0.35,
    causal_anomaly_drivers: bool = False,
    absolute_state_targets: bool = False,
):
    """Make slow physical exposure/recovery the primary quality mechanism.

    Truth-weather loading drives the physical state; a parallel noisy-nowcast
    recurrence supplies the forecast-quality diagnostic. The scheduler sees
    only current reportable quality and ordinary nowcasts, never these state
    columns. This mode intentionally replaces, rather than multiplies, the
    fast weather-exposure quality profile.
    """
    out = frame.copy()
    required = (
        "wind_speed_ms", "relative_humidity", "air_temperature_c",
        "solar_radiation_wm2", "agent_context_nowcast_wind_speed_ms",
        "agent_context_nowcast_relative_humidity",
        "agent_context_nowcast_air_temperature_c",
        "agent_context_nowcast_solar_radiation_wm2",
    )
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"exposure/recovery state requires columns: {missing}")

    def unit(values: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(values, [0.05, 0.95])
        return np.clip((values - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)

    def causal_anomaly(values: np.ndarray, scale: float, *, invert: bool = False) -> np.ndarray:
        baseline = np.empty_like(values, dtype=float)
        baseline[0] = float(values[0])
        alpha = 1.0 / 168.0
        for index in range(1, len(values)):
            baseline[index] = (1.0 - alpha) * baseline[index - 1] + alpha * float(values[index])
        anomaly = (values - baseline) / max(float(scale), 1.0e-6)
        if invert:
            anomaly = -anomaly
        return 1.0 / (1.0 + np.exp(-np.clip(anomaly, -12.0, 12.0)))

    def states(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        wind_values = out[f"{prefix}wind_speed_ms"].to_numpy(dtype=float)
        humidity_values = out[f"{prefix}relative_humidity"].to_numpy(dtype=float)
        temperature_values = out[f"{prefix}air_temperature_c"].to_numpy(dtype=float)
        solar_values = out[f"{prefix}solar_radiation_wm2"].to_numpy(dtype=float)
        if causal_anomaly_drivers:
            wind = causal_anomaly(wind_values, 3.0)
            humidity = causal_anomaly(humidity_values, 8.0)
            cold = causal_anomaly(temperature_values, 8.0, invert=True)
            low_solar = causal_anomaly(solar_values, 100.0, invert=True)
        else:
            wind = unit(wind_values)
            humidity = unit(humidity_values)
            cold = unit(-temperature_values)
            low_solar = 1.0 - unit(solar_values)
        if decoupled_drivers:
            loading = np.clip(0.85 * wind + 0.10 * humidity + 0.05 * cold, 0.0, 1.0)
        else:
            loading = np.clip(0.60 * wind + 0.25 * humidity + 0.15 * cold, 0.0, 1.0)
        frost_loading = np.clip(0.55 * humidity + 0.30 * cold + 0.15 * low_solar, 0.0, 1.0)
        if decoupled_drivers:
            transport_recovery = np.clip(0.75 * (1.0 - wind) + 0.25 * (1.0 - cold), 0.0, 1.0)
            frost_recovery = np.clip(0.60 * (1.0 - cold) + 0.40 * (1.0 - low_solar), 0.0, 1.0)
        else:
            recovery = np.clip(0.65 * (1.0 - cold) + 0.35 * (1.0 - low_solar), 0.0, 1.0)
            transport_recovery = recovery
            frost_recovery = recovery
        transport = np.empty_like(loading)
        frost = np.empty_like(loading)
        transport[0] = loading[0]
        frost[0] = frost_loading[0]
        for idx in range(1, len(loading)):
            if balanced_recovery:
                previous_transport = transport[idx - 1]
                previous_frost = frost[idx - 1]
                transport[idx] = np.clip(
                    previous_transport
                    + 0.08 * loading[idx] * (1.0 - previous_transport)
                    - 0.10 * transport_recovery[idx] * previous_transport,
                    0.0,
                    1.0,
                )
                frost[idx] = np.clip(
                    previous_frost
                    + 0.06 * frost_loading[idx] * (1.0 - previous_frost)
                    - 0.12 * frost_recovery[idx] * previous_frost,
                    0.0,
                    1.0,
                )
            else:
                transport[idx] = np.clip(0.94 * transport[idx - 1] + 0.10 * loading[idx] - 0.04 * transport_recovery[idx], 0.0, 1.0)
                frost[idx] = np.clip(0.96 * frost[idx - 1] + 0.08 * frost_loading[idx] - 0.05 * frost_recovery[idx], 0.0, 1.0)
        return transport, frost

    transport, frost = states("")
    forecast_transport, forecast_frost = states("agent_context_nowcast_")
    actual_quality = {
        "met_station_core": 0.96 - 0.16 * transport - 0.08 * frost,
        "radiometer_basic": 0.96 - 0.10 * transport - 0.38 * frost,
        "shielded_thermo_hygro": 0.97 - 0.05 * transport - 0.10 * frost,
        "surface_temp_ir": 0.96 - 0.08 * transport - 0.48 * frost,
        "laser_disdrometer": 0.70 + 0.28 * transport - 0.22 * frost,
        "fc4_flux": 0.62 + 0.34 * transport - 0.16 * frost,
    }
    forecast_quality = {
        key: value for key, value in actual_quality.items()
    }
    # Re-evaluate the identical fixed response curves on forecast states.
    for sensor_id in sensor_ids:
        key = str(sensor_id)
        if key not in actual_quality:
            continue
        expression = {
            "met_station_core": lambda t, f: 0.96 - 0.16 * t - 0.08 * f,
            "radiometer_basic": lambda t, f: 0.96 - 0.10 * t - 0.38 * f,
            "shielded_thermo_hygro": lambda t, f: 0.97 - 0.05 * t - 0.10 * f,
            "surface_temp_ir": lambda t, f: 0.96 - 0.08 * t - 0.48 * f,
            "laser_disdrometer": lambda t, f: 0.70 + 0.28 * t - 0.22 * f,
            "fc4_flux": lambda t, f: 0.62 + 0.34 * t - 0.16 * f,
        }[key]
        out[f"agent_context_quality_{key}"] = np.clip(actual_quality[key], 0.10, 1.0)
        out[f"agent_context_quality_forecast_{key}"] = np.clip(expression(forecast_transport, forecast_frost), 0.10, 1.0)

    # Lagged target consequences remain bounded and do not encode a channel ID.
    gain = max(0.0, float(target_gain))
    if low_frequency_targets or absolute_state_targets:
        event_column = "blowing_snow_active" if "blowing_snow_active" in out else "event_flag"
        if event_column not in out:
            raise ValueError("low-frequency exposure targets require a blowing-snow activity column")
        active = out[event_column].to_numpy(dtype=bool)
        residual_weight = float(np.clip(residual_fraction, 0.0, 1.0))

        def causal_ema(values: np.ndarray, alpha: float = 1.0 / 12.0) -> np.ndarray:
            baseline = np.empty_like(values, dtype=float)
            baseline[0] = float(values[0])
            for index in range(1, len(values)):
                baseline[index] = (1.0 - alpha) * baseline[index - 1] + alpha * values[index]
            return baseline

        original_flux = out["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float)
        original_velocity = out["snow_particle_mean_velocity_ms"].to_numpy(dtype=float)
        original_diameter = out["snow_particle_mean_diameter_mm"].to_numpy(dtype=float)
        if absolute_state_targets:
            def causal_innovation(values: np.ndarray, initial_scale: float) -> np.ndarray:
                mean = float(values[0])
                deviation = max(float(initial_scale), 1.0e-9)
                alpha = 1.0 / 12.0
                result = np.zeros_like(values, dtype=float)
                for index, value in enumerate(values):
                    delta = float(value) - mean
                    result[index] = np.clip(delta / max(deviation, initial_scale), -3.0, 3.0)
                    mean = (1.0 - alpha) * mean + alpha * float(value)
                    deviation = (1.0 - alpha) * deviation + alpha * abs(delta)
                return result

            flux_innovation = causal_innovation(original_flux, 5.0e-6)
            velocity_innovation = causal_innovation(original_velocity, 2.0)
            diameter_innovation = causal_innovation(original_diameter, 0.05)
            transformed_flux = np.clip(
                (1.0e-6 + 4.0e-5 * np.square(transport))
                * np.exp(0.18 * flux_innovation),
                0.0,
                None,
            )
            transformed_velocity = np.clip(
                1.5 + 8.0 * transport + 1.2 * velocity_innovation,
                0.0,
                20.0,
            )
            transformed_diameter = np.clip(
                0.07 + 0.18 * transport + 0.08 * (1.0 - frost)
                + 0.025 * diameter_innovation,
                0.04,
                0.5,
            )
        else:
            flux_base = causal_ema(original_flux)
            velocity_base = causal_ema(original_velocity)
            diameter_base = causal_ema(original_diameter)
            transformed_flux = np.clip(
                flux_base * np.exp(0.90 * (transport - 0.5))
                + residual_weight * (original_flux - flux_base),
                0.0,
                None,
            )
            transformed_velocity = np.clip(
                velocity_base + 2.40 * (transport - 0.5)
                + residual_weight * (original_velocity - velocity_base),
                0.0,
                20.0,
            )
            transformed_diameter = np.clip(
                diameter_base + 0.070 * (transport - 0.5)
                + residual_weight * (original_diameter - diameter_base),
                0.04,
                0.5,
            )
        out["snow_mass_flux_kg_m2_s"] = np.where(active, transformed_flux, original_flux)
        out["snow_particle_mean_velocity_ms"] = np.where(active, transformed_velocity, original_velocity)
        out["snow_particle_mean_diameter_mm"] = np.where(active, transformed_diameter, original_diameter)
    else:
        out["snow_mass_flux_kg_m2_s"] *= np.exp(0.45 * gain * (transport - 0.5))
        out["snow_particle_mean_velocity_ms"] = np.clip(out["snow_particle_mean_velocity_ms"] + 1.10 * gain * (transport - 0.5), 0.0, 20.0)
        out["snow_particle_mean_diameter_mm"] = np.clip(out["snow_particle_mean_diameter_mm"] + 0.030 * gain * (transport - 0.5), 0.04, 0.5)
    out["snow_surface_temperature_c"] += 1.20 * (frost - 0.5)
    out["generator_exposure_transport_state"] = transport
    out["generator_exposure_frost_state"] = frost
    return out


def add_three_factor_exposure_dynamics(
    frame,
    *,
    sensor_ids: tuple[str, ...],
    faster_thermal_response: bool = False,
    dual_timescale_thermal_target: bool = False,
):
    """Apply causal bulk-flux, particle-loading, and thermal exposure states."""
    out = frame.copy()
    required = (
        "wind_speed_ms", "relative_humidity", "air_temperature_c",
        "solar_radiation_wm2", "agent_context_nowcast_wind_speed_ms",
        "agent_context_nowcast_relative_humidity",
        "agent_context_nowcast_air_temperature_c",
        "agent_context_nowcast_solar_radiation_wm2",
    )
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"three-factor exposure state requires columns: {missing}")

    def anomaly(values: np.ndarray, scale: float, *, invert: bool = False) -> np.ndarray:
        baseline = np.empty_like(values, dtype=float)
        baseline[0] = float(values[0])
        alpha = 1.0 / 168.0
        for index in range(1, len(values)):
            baseline[index] = (1.0 - alpha) * baseline[index - 1] + alpha * float(values[index])
        z = (values - baseline) / max(float(scale), 1.0e-6)
        if invert:
            z = -z
        return 1.0 / (1.0 + np.exp(-np.clip(z, -12.0, 12.0)))

    def recurrence(loading: np.ndarray, *, load_rate: float, recover_rate: float) -> np.ndarray:
        state = np.empty_like(loading, dtype=float)
        state[0] = float(loading[0])
        for index in range(1, len(loading)):
            previous = state[index - 1]
            state[index] = np.clip(
                previous
                + load_rate * loading[index] * (1.0 - previous)
                - recover_rate * (1.0 - loading[index]) * previous,
                0.0,
                1.0,
            )
        return state

    def build_states(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        wind = anomaly(out[f"{prefix}wind_speed_ms"].to_numpy(dtype=float), 3.0)
        humidity = anomaly(out[f"{prefix}relative_humidity"].to_numpy(dtype=float), 8.0)
        cold = anomaly(out[f"{prefix}air_temperature_c"].to_numpy(dtype=float), 8.0, invert=True)
        low_solar = anomaly(out[f"{prefix}solar_radiation_wm2"].to_numpy(dtype=float), 100.0, invert=True)
        thermal_loading = np.clip(0.65 * cold + 0.35 * low_solar, 0.0, 1.0)
        thermal_load_rate, thermal_recover_rate = (
            (0.10, 0.16) if faster_thermal_response else (0.06, 0.12)
        )
        return (
            recurrence(wind, load_rate=0.08, recover_rate=0.10),
            recurrence(humidity, load_rate=0.07, recover_rate=0.11),
            recurrence(thermal_loading, load_rate=thermal_load_rate, recover_rate=thermal_recover_rate),
            thermal_loading,
        )

    flux_state, particle_state, thermal_state, thermal_loading = build_states("")
    forecast_flux, forecast_particle, forecast_thermal, _ = build_states("agent_context_nowcast_")

    def quality_values(flux: np.ndarray, particle: np.ndarray, thermal: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "met_station_core": 0.95 - 0.18 * flux - 0.08 * thermal,
            "radiometer_basic": 0.98 - 0.48 * thermal,
            "shielded_thermo_hygro": 0.94 - 0.18 * thermal - 0.08 * particle,
            "surface_temp_ir": 0.98 - 0.58 * thermal,
            "laser_disdrometer": 0.96 - 0.42 * particle - 0.12 * thermal,
            "fc4_flux": 0.96 - 0.38 * flux - 0.10 * particle,
        }

    actual_quality = quality_values(flux_state, particle_state, thermal_state)
    forecast_quality = quality_values(forecast_flux, forecast_particle, forecast_thermal)
    for sensor_id in sensor_ids:
        key = str(sensor_id)
        if key in actual_quality:
            out[f"agent_context_quality_{key}"] = np.clip(actual_quality[key], 0.10, 1.0)
            out[f"agent_context_quality_forecast_{key}"] = np.clip(forecast_quality[key], 0.10, 1.0)

    active_column = "blowing_snow_active" if "blowing_snow_active" in out else "event_flag"
    active = out[active_column].to_numpy(dtype=bool)

    def innovation(values: np.ndarray, initial_scale: float) -> np.ndarray:
        mean = float(values[0])
        deviation = max(float(initial_scale), 1.0e-9)
        alpha = 1.0 / 12.0
        result = np.zeros_like(values, dtype=float)
        for index, value in enumerate(values):
            delta = float(value) - mean
            result[index] = np.clip(delta / max(deviation, initial_scale), -3.0, 3.0)
            mean = (1.0 - alpha) * mean + alpha * float(value)
            deviation = (1.0 - alpha) * deviation + alpha * abs(delta)
        return result

    original_flux = out["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float)
    original_velocity = out["snow_particle_mean_velocity_ms"].to_numpy(dtype=float)
    original_diameter = out["snow_particle_mean_diameter_mm"].to_numpy(dtype=float)
    original_surface = out["snow_surface_temperature_c"].to_numpy(dtype=float)
    flux = np.clip((1.0e-6 + 4.5e-5 * np.square(flux_state)) * np.exp(0.16 * innovation(original_flux, 5.0e-6)), 0.0, None)
    velocity = np.clip(1.5 + 7.0 * particle_state + 1.0 * innovation(original_velocity, 2.0), 0.0, 20.0)
    diameter = np.clip(0.07 + 0.22 * particle_state + 0.04 * (1.0 - thermal_state) + 0.020 * innovation(original_diameter, 0.05), 0.04, 0.5)
    surface_innovation_scale = 1.5 if faster_thermal_response else 0.8
    thermal_gap = (
        1.0 + 4.0 * thermal_state + 2.0 * thermal_loading
        if dual_timescale_thermal_target
        else 1.0 + 6.0 * thermal_state
    )
    surface = np.clip(
        out["air_temperature_c"].to_numpy(dtype=float) - thermal_gap
        + surface_innovation_scale * innovation(original_surface, 2.0),
        -80.0,
        10.0,
    )
    out["snow_mass_flux_kg_m2_s"] = np.where(active, flux, original_flux)
    out["snow_particle_mean_velocity_ms"] = np.where(active, velocity, original_velocity)
    out["snow_particle_mean_diameter_mm"] = np.where(active, diameter, original_diameter)
    out["snow_surface_temperature_c"] = surface
    out["generator_flux_exposure_state"] = flux_state
    out["generator_particle_exposure_state"] = particle_state
    out["generator_thermal_exposure_state"] = thermal_state
    return out


def add_forecast_value_dynamics(
    frame,
    *,
    sensor_ids: tuple[str, ...],
    seed: int,
    stationary_local_state: bool = False,
    residence_local_state: bool = False,
    horizon_persistent_latent: bool = False,
    specialist_resilient_quality: bool = False,
    activity_aligned_transport_demand: bool = False,
    forecast_lead_steps: int = 8,
):
    """Add forecastable demand and persistent unresolved target components."""
    out = frame.copy()
    required = (
        "wind_speed_ms", "relative_humidity", "air_temperature_c",
        "solar_radiation_wm2", "agent_context_nowcast_wind_speed_ms",
        "agent_context_nowcast_relative_humidity",
        "agent_context_nowcast_air_temperature_c",
        "agent_context_nowcast_solar_radiation_wm2",
        "snow_mass_flux_kg_m2_s", "snow_particle_mean_velocity_ms",
        "snow_particle_mean_diameter_mm", "snow_surface_temperature_c",
    )
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"forecast-value dynamics require columns: {missing}")

    def anomaly(values: np.ndarray, scale: float, *, invert: bool = False) -> np.ndarray:
        baseline = np.empty_like(values, dtype=float)
        baseline[0] = float(values[0])
        alpha = 1.0 / 168.0
        for index in range(1, len(values)):
            baseline[index] = (1.0 - alpha) * baseline[index - 1] + alpha * float(values[index])
        result = (values - baseline) / max(float(scale), 1.0e-6)
        if invert:
            result = -result
        return 1.0 / (1.0 + np.exp(-np.clip(result, -12.0, 12.0)))

    def recurrence(loading: np.ndarray, load_rate: float, recover_rate: float) -> np.ndarray:
        state = np.empty_like(loading, dtype=float)
        state[0] = float(loading[0])
        for index in range(1, len(loading)):
            previous = state[index - 1]
            state[index] = np.clip(
                previous
                + load_rate * loading[index] * (1.0 - previous)
                - recover_rate * (1.0 - loading[index]) * previous,
                0.0,
                1.0,
            )
        return state

    def states(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wind = anomaly(out[f"{prefix}wind_speed_ms"].to_numpy(dtype=float), 3.0)
        humidity = anomaly(out[f"{prefix}relative_humidity"].to_numpy(dtype=float), 8.0)
        cold = anomaly(out[f"{prefix}air_temperature_c"].to_numpy(dtype=float), 8.0, invert=True)
        low_solar = anomaly(out[f"{prefix}solar_radiation_wm2"].to_numpy(dtype=float), 100.0, invert=True)
        thermal = np.clip(0.65 * cold + 0.35 * low_solar, 0.0, 1.0)
        return (
            recurrence(wind, 0.08, 0.10),
            recurrence(humidity, 0.07, 0.11),
            recurrence(thermal, 0.10, 0.16),
        )

    if stationary_local_state and residence_local_state:
        raise ValueError("stationary and residence local states are mutually exclusive")
    if activity_aligned_transport_demand and not residence_local_state:
        raise ValueError("activity-aligned transport demand requires residence local state")

    active_column = "blowing_snow_active" if "blowing_snow_active" in out else "event_flag"
    if active_column not in out:
        raise ValueError("forecast-value dynamics require a blowing-snow activity column")
    active = out[active_column].to_numpy(dtype=bool)

    if stationary_local_state or residence_local_state:
        weather_states = states("")

        def stationary_state(stream: int, loading: np.ndarray) -> np.ndarray:
            rho = 0.965
            rng = np.random.default_rng(int(seed) + int(stream))
            noise = rng.normal(0.0, np.sqrt(1.0 - rho * rho), size=len(out))
            latent = np.empty(len(out), dtype=float)
            latent[0] = float(noise[0])
            for index in range(1, len(latent)):
                latent[index] = (
                    rho * latent[index - 1]
                    + noise[index]
                    + 0.08 * (float(loading[index]) - 0.5)
                )
            return 1.0 / (1.0 + np.exp(-np.clip(latent, -12.0, 12.0)))

        def residence_state(stream: int, loading: np.ndarray) -> np.ndarray:
            rng = np.random.default_rng(int(seed) + int(stream))
            values = np.empty(len(out), dtype=float)
            high = bool(rng.integers(0, 2))
            position = 0
            perturbation = 0.0
            while position < len(values):
                duration = int(rng.integers(18, 55))
                centre = float(rng.uniform(0.70, 0.84) if high else rng.uniform(0.16, 0.30))
                stop = min(len(values), position + duration)
                for index in range(position, stop):
                    perturbation = 0.88 * perturbation + float(rng.normal(0.0, 0.018))
                    values[index] = np.clip(
                        centre + perturbation + 0.04 * (float(loading[index]) - 0.5),
                        0.05,
                        0.95,
                    )
                position = stop
                high = not high
            return values

        state_builder = residence_state if residence_local_state else stationary_state
        raw_flux = state_builder(110_003, weather_states[0])
        raw_particle = state_builder(110_017, weather_states[1])
        actual_thermal = state_builder(110_029, weather_states[2])

        if activity_aligned_transport_demand:
            activity = active.astype(float)
            actual_flux = raw_flux * activity
            actual_particle = raw_particle * activity
            out["generator_flux_demand_state_raw"] = raw_flux
            out["generator_particle_demand_state_raw"] = raw_particle
        else:
            actual_flux = raw_flux
            actual_particle = raw_particle

        def forecast_state(stream: int, actual: np.ndarray) -> np.ndarray:
            lead = max(0, int(forecast_lead_steps))
            future = np.empty_like(actual)
            if lead > 0:
                future[:-lead] = actual[lead:]
                future[-lead:] = actual[-1]
            else:
                future[:] = actual
            rng = np.random.default_rng(int(seed) + int(stream))
            return np.clip(future + rng.normal(0.0, 0.08, size=len(actual)), 0.0, 1.0)

        forecast_flux = forecast_state(115_003, actual_flux)
        forecast_particle = forecast_state(115_017, actual_particle)
        forecast_thermal = forecast_state(115_029, actual_thermal)
    else:
        actual_flux, actual_particle, actual_thermal = states("")
        forecast_flux, forecast_particle, forecast_thermal = states("agent_context_nowcast_")
    for name, actual, forecast in (
        ("flux", actual_flux, forecast_flux),
        ("particle", actual_particle, forecast_particle),
        ("thermal", actual_thermal, forecast_thermal),
    ):
        out[f"generator_{name}_demand_state"] = actual
        out[f"agent_context_forecast_{name}_demand"] = forecast

    def ar1(stream: int) -> np.ndarray:
        rho = 0.97 if horizon_persistent_latent else 0.92
        rng = np.random.default_rng(int(seed) + int(stream))
        innovations = rng.normal(0.0, np.sqrt(1.0 - rho * rho), size=len(out))
        values = np.empty(len(out), dtype=float)
        values[0] = float(innovations[0])
        for index in range(1, len(values)):
            values[index] = rho * values[index - 1] + innovations[index]
        return values

    flux_latent = ar1(120_011)
    particle_latent = ar1(120_023)
    thermal_latent = ar1(120_037)
    if stationary_local_state or residence_local_state:
        flux_amplitude = 0.10 + 0.90 * np.square(actual_flux)
        particle_amplitude = 0.10 + 0.90 * np.square(actual_particle)
        thermal_amplitude = 0.10 + 0.90 * np.square(actual_thermal)
    else:
        flux_amplitude = 0.20 + 0.80 * actual_flux
        particle_amplitude = 0.20 + 0.80 * actual_particle
        thermal_amplitude = 0.20 + 0.80 * actual_thermal
    base_flux = np.clip(1.0e-6 + 4.5e-5 * np.square(actual_flux), 0.0, None)
    base_velocity = 1.5 + 7.0 * actual_particle
    base_diameter = 0.07 + 0.22 * actual_particle + 0.04 * (1.0 - actual_thermal)
    base_surface = out["air_temperature_c"].to_numpy(dtype=float) - (1.0 + 5.0 * actual_thermal)
    out["snow_mass_flux_kg_m2_s"] = np.where(
        active,
        base_flux * np.exp(0.45 * flux_amplitude * flux_latent),
        out["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float),
    )
    out["snow_particle_mean_velocity_ms"] = np.where(
        active,
        np.clip(base_velocity + 2.20 * particle_amplitude * particle_latent, 0.0, 20.0),
        out["snow_particle_mean_velocity_ms"].to_numpy(dtype=float),
    )
    out["snow_particle_mean_diameter_mm"] = np.where(
        active,
        np.clip(base_diameter + 0.055 * particle_amplitude * particle_latent, 0.04, 0.5),
        out["snow_particle_mean_diameter_mm"].to_numpy(dtype=float),
    )
    out["snow_surface_temperature_c"] = np.clip(
        base_surface + 2.20 * thermal_amplitude * thermal_latent,
        -80.0,
        10.0,
    )

    def quality_values(flux: np.ndarray, particle: np.ndarray, thermal: np.ndarray) -> dict[str, np.ndarray]:
        if specialist_resilient_quality:
            return {
                "met_station_core": 0.97 - 0.10 * flux - 0.05 * thermal,
                "radiometer_basic": 0.98 - 0.16 * thermal,
                "shielded_thermo_hygro": 0.97 - 0.08 * particle - 0.05 * thermal,
                "surface_temp_ir": 0.97 - 0.03 * particle,
                "laser_disdrometer": 0.97 - 0.03 * thermal,
                "fc4_flux": 0.97 - 0.03 * particle,
            }
        return {
            "met_station_core": 0.97 - 0.10 * flux - 0.05 * thermal,
            "radiometer_basic": 0.98 - 0.16 * thermal,
            "shielded_thermo_hygro": 0.97 - 0.08 * particle - 0.05 * thermal,
            "surface_temp_ir": 0.98 - 0.20 * thermal,
            "laser_disdrometer": 0.98 - 0.20 * particle - 0.04 * thermal,
            "fc4_flux": 0.98 - 0.20 * flux - 0.04 * particle,
        }

    actual_quality = quality_values(actual_flux, actual_particle, actual_thermal)
    forecast_quality = quality_values(forecast_flux, forecast_particle, forecast_thermal)
    for sensor_id in sensor_ids:
        key = str(sensor_id)
        if key in actual_quality:
            out[f"agent_context_quality_{key}"] = np.clip(actual_quality[key], 0.72, 1.0)
            out[f"agent_context_quality_forecast_{key}"] = np.clip(forecast_quality[key], 0.72, 1.0)
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
        choices=["random", "stratified", "stratified_duration", "cycling"],
        default="random",
    )
    parser.add_argument("--event-subtype-cycle-steps", type=int, default=0)
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
    parser.add_argument("--nowcast-solar-noise-std", type=float, default=35.0)
    parser.add_argument("--channel-quality-enabled", action="store_true")
    parser.add_argument(
        "--channel-quality-mode",
        choices=[
            "independent", "condition_dependent", "condition_dependent_crossover",
            "condition_dependent_crossover_strong",
            "condition_dependent_crossover_balanced",
            "condition_dependent_crossover_calibrated",
            "condition_dependent_crossover_robust",
        ],
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
    parser.add_argument(
        "--continuous-operating-state",
        action="store_true",
        help="Apply persistent target/reliability factors driven only by noisy nowcasts.",
    )
    parser.add_argument("--operating-transport-rho", type=float, default=0.90)
    parser.add_argument("--operating-thermal-rho", type=float, default=0.94)
    parser.add_argument("--operating-target-scale", type=float, default=1.0)
    parser.add_argument("--operating-quality-scale", type=float, default=1.0)
    parser.add_argument("--exposure-recovery-state", action="store_true")
    parser.add_argument("--balanced-exposure-recovery-state", action="store_true")
    parser.add_argument("--decoupled-exposure-recovery-state", action="store_true")
    parser.add_argument("--exposure-target-gain", type=float, default=1.0)
    parser.add_argument("--exposure-low-frequency-targets", action="store_true")
    parser.add_argument("--exposure-residual-fraction", type=float, default=0.35)
    parser.add_argument("--exposure-causal-anomaly-drivers", action="store_true")
    parser.add_argument("--exposure-absolute-state-targets", action="store_true")
    parser.add_argument("--three-factor-exposure-state", action="store_true")
    parser.add_argument("--three-factor-faster-thermal-response", action="store_true")
    parser.add_argument("--three-factor-dual-timescale-thermal-target", action="store_true")
    parser.add_argument("--forecast-value-state", action="store_true")
    parser.add_argument("--forecast-value-stationary-local-state", action="store_true")
    parser.add_argument("--forecast-value-residence-local-state", action="store_true")
    parser.add_argument("--forecast-value-horizon-persistent-latent", action="store_true")
    parser.add_argument("--forecast-value-specialist-resilient-quality", action="store_true")
    parser.add_argument("--forecast-value-activity-aligned-transport-demand", action="store_true")
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
        event_subtype_cycle_steps=int(args.event_subtype_cycle_steps),
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
        nowcast_solar_noise_std=float(args.nowcast_solar_noise_std),
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
            forecast_quality=bool(int(args.nowcast_lead_steps) > 0),
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
            "forecast_quality_enabled": bool(int(args.nowcast_lead_steps) > 0),
        }
    if bool(args.continuous_operating_state):
        df = add_nowcast_operating_state_dynamics(
            df,
            sensor_ids=tuple(str(value) for value in args.channel_quality_sensor_ids),
            transport_rho=float(args.operating_transport_rho),
            thermal_rho=float(args.operating_thermal_rho),
            target_scale=float(args.operating_target_scale),
            quality_scale=float(args.operating_quality_scale),
        )
        meta["continuous_operating_state"] = {
            "enabled": True,
            "driver": "noisy_nowcasts_only",
            "transport_rho": float(args.operating_transport_rho),
            "thermal_rho": float(args.operating_thermal_rho),
            "target_scale": float(args.operating_target_scale),
            "quality_scale": float(args.operating_quality_scale),
            "diagnostic_columns": [
                "generator_operating_transport_state",
                "generator_operating_thermal_state",
            ],
        }
    if any((args.exposure_recovery_state, args.balanced_exposure_recovery_state, args.decoupled_exposure_recovery_state)):
        if bool(args.continuous_operating_state):
            raise ValueError("exposure/recovery and continuous operating state are mutually exclusive")
        if sum(bool(value) for value in (
            args.exposure_recovery_state,
            args.balanced_exposure_recovery_state,
            args.decoupled_exposure_recovery_state,
        )) != 1:
            raise ValueError("legacy, balanced, and decoupled exposure/recovery modes are mutually exclusive")
        df = add_exposure_recovery_dynamics(
            df,
            sensor_ids=tuple(str(value) for value in args.channel_quality_sensor_ids),
            balanced_recovery=bool(args.balanced_exposure_recovery_state or args.decoupled_exposure_recovery_state),
            decoupled_drivers=bool(args.decoupled_exposure_recovery_state),
            target_gain=float(args.exposure_target_gain),
            low_frequency_targets=bool(args.exposure_low_frequency_targets),
            residual_fraction=float(args.exposure_residual_fraction),
            causal_anomaly_drivers=bool(args.exposure_causal_anomaly_drivers),
            absolute_state_targets=bool(args.exposure_absolute_state_targets),
        )
        meta["exposure_recovery_state"] = {
            "enabled": True,
            "mode": (
                "balanced_decoupled"
                if bool(args.decoupled_exposure_recovery_state)
                else "balanced_state_proportional"
                if bool(args.balanced_exposure_recovery_state)
                else "legacy_additive"
            ),
            "truth_driver": "weather_loading_recovery",
            "forecast_driver": "noisy_nowcast_loading_recovery",
            "target_gain": float(args.exposure_target_gain),
            "low_frequency_targets": bool(args.exposure_low_frequency_targets),
            "residual_fraction": float(args.exposure_residual_fraction),
            "causal_anomaly_drivers": bool(args.exposure_causal_anomaly_drivers),
            "absolute_state_targets": bool(args.exposure_absolute_state_targets),
            "diagnostic_columns": ["generator_exposure_transport_state", "generator_exposure_frost_state"],
        }
    if bool(args.three_factor_exposure_state):
        if any((args.continuous_operating_state, args.exposure_recovery_state, args.balanced_exposure_recovery_state, args.decoupled_exposure_recovery_state, args.forecast_value_state)):
            raise ValueError("three-factor exposure state is mutually exclusive with other operating-state modes")
        df = add_three_factor_exposure_dynamics(
            df,
            sensor_ids=tuple(str(value) for value in args.channel_quality_sensor_ids),
            faster_thermal_response=bool(args.three_factor_faster_thermal_response),
            dual_timescale_thermal_target=bool(args.three_factor_dual_timescale_thermal_target),
        )
        meta["three_factor_exposure_state"] = {
            "enabled": True,
            "driver": "causal_weather_anomalies",
            "online_forecast_driver": "causal_noisy_nowcast_anomalies",
            "faster_thermal_response": bool(args.three_factor_faster_thermal_response),
            "dual_timescale_thermal_target": bool(args.three_factor_dual_timescale_thermal_target),
            "diagnostic_columns": [
                "generator_flux_exposure_state",
                "generator_particle_exposure_state",
                "generator_thermal_exposure_state",
            ],
        }
    if bool(args.forecast_value_state):
        if any((args.continuous_operating_state, args.exposure_recovery_state, args.balanced_exposure_recovery_state, args.decoupled_exposure_recovery_state, args.three_factor_exposure_state)):
            raise ValueError("forecast-value state is mutually exclusive with other operating-state modes")
        df = add_forecast_value_dynamics(
            df,
            sensor_ids=tuple(str(value) for value in args.channel_quality_sensor_ids),
            seed=int(args.seed),
            stationary_local_state=bool(args.forecast_value_stationary_local_state),
            residence_local_state=bool(args.forecast_value_residence_local_state),
            horizon_persistent_latent=bool(args.forecast_value_horizon_persistent_latent),
            specialist_resilient_quality=bool(args.forecast_value_specialist_resilient_quality),
            activity_aligned_transport_demand=bool(args.forecast_value_activity_aligned_transport_demand),
            forecast_lead_steps=int(args.nowcast_lead_steps),
        )
        meta["forecast_value_state"] = {
            "enabled": True,
            "mechanism": "forecastable_demand_with_persistent_unresolved_target_component",
            "latent_rho": 0.97 if bool(args.forecast_value_horizon_persistent_latent) else 0.92,
            "amplitude_floor": 0.10 if bool(args.forecast_value_stationary_local_state or args.forecast_value_residence_local_state) else 0.20,
            "amplitude_mapping": (
                "0.10_plus_0.90_state_squared"
                if bool(args.forecast_value_stationary_local_state or args.forecast_value_residence_local_state)
                else "0.20_plus_0.80_state"
            ),
            "stationary_local_state": bool(args.forecast_value_stationary_local_state),
            "residence_local_state": bool(args.forecast_value_residence_local_state),
            "horizon_persistent_latent": bool(args.forecast_value_horizon_persistent_latent),
            "specialist_resilient_quality": bool(args.forecast_value_specialist_resilient_quality),
            "activity_aligned_transport_demand": bool(args.forecast_value_activity_aligned_transport_demand),
            "online_columns": [
                "agent_context_forecast_flux_demand",
                "agent_context_forecast_particle_demand",
                "agent_context_forecast_thermal_demand",
            ],
            "diagnostic_columns": [
                "generator_flux_demand_state",
                "generator_particle_demand_state",
                "generator_thermal_demand_state",
            ] + (
                ["generator_flux_demand_state_raw", "generator_particle_demand_state_raw"]
                if bool(args.forecast_value_activity_aligned_transport_demand)
                else []
            ),
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
