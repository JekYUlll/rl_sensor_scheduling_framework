from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats


STATE_COLUMNS = [
    "wind_speed_ms",
    "wind_direction_deg",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
    "solar_radiation_wm2",
    "snow_surface_temperature_c",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
    "snow_mass_flux_kg_m2_s",
]


def _canonical_antaws_column(raw_name: str) -> str | None:
    normalized = raw_name.strip().lower()
    if normalized.startswith("temperature"):
        return "air_temperature_c"
    if normalized.startswith("pressure"):
        return "air_pressure_hpa"
    if normalized.startswith("wind speed"):
        return "wind_speed_ms"
    if normalized.startswith("wind direction"):
        return "wind_direction_deg"
    if normalized.startswith("relative humidity"):
        return "relative_humidity"
    return None


@dataclass(frozen=True)
class PublicWeatherSynthesisConfig:
    antaws_root: Path
    stations: tuple[str, ...] = ("Panda100", "Panda200", "Taishan")
    steps: int = 10000
    freq_s: int = 3600
    seed: int = 42
    phase_keep_fraction: float = 0.15
    match_marginal_distribution: bool = True
    wind_threshold_ms: float = 8.0
    strong_wind_threshold_ms: float = 12.0
    parsivel_min_temp_c: float = -30.0
    blowing_snow_event_coverage: float = 0.30
    blowing_snow_event_model: str = "clustered"
    blowing_snow_min_duration_steps: int = 10
    blowing_snow_max_duration_steps: int = 30
    blowing_snow_min_gap_steps: int = 6
    blowing_snow_lead_steps: int = 5
    blowing_snow_wind_margin_ms: float = 1.5
    cred_hysteresis_on: float = 0.6
    cred_hysteresis_off: float = 0.3
    flux_wind_exponent: float = 3.6
    event_microstructure_sigma: float = 0.0
    event_microstructure_alpha: float = 0.18
    event_microstructure_diameter_scale: float = 0.0
    event_microstructure_velocity_scale: float = 0.0
    event_particle_microstructure_correlation: float = 1.0
    event_subtypes_enabled: bool = False
    event_subtype_assignment: str = "random"
    event_subtype_particle_min_parsivel_availability: float = 0.0
    event_subtype_particle_prob: float = 0.34
    event_subtype_flux_prob: float = 0.33
    event_subtype_thermal_prob: float = 0.33
    event_subtype_particle_flux_multiplier: float = 0.72
    event_subtype_flux_multiplier: float = 2.4
    event_subtype_thermal_flux_multiplier: float = 0.55
    event_subtype_particle_diameter_shift_mm: float = 0.10
    event_subtype_particle_velocity_boost_ms: float = 1.3
    event_subtype_flux_diameter_shift_mm: float = -0.04
    event_subtype_flux_velocity_boost_ms: float = 0.7
    event_subtype_thermal_surface_drop_c: float = 2.0
    event_subtype_particle_humidity_boost_pct: float = 0.0
    event_subtype_flux_wind_boost_ms: float = 0.0
    event_subtype_thermal_air_temp_drop_c: float = 0.0
    event_subtype_latent_alpha: float = 0.18
    event_subtype_particle_latent_diameter_scale_mm: float = 0.0
    event_subtype_particle_latent_velocity_scale_ms: float = 0.0
    event_subtype_flux_latent_sigma: float = 0.0
    event_subtype_flux_latent_linear_scale: float = 0.0
    event_subtype_flux_latent_linear_offset: float = 1.5
    event_subtype_flux_latent_linear_clip: float = 4.0
    event_subtype_thermal_latent_surface_scale_c: float = 0.0
    event_subtype_latent_target_lag_steps: int = 0
    event_subtype_context_lead_steps: int = 0
    event_subtype_context_noise_std: float = 0.08
    event_subtype_context_latent_strength: float = 0.0


def load_antaws_station(antaws_root: str | Path, station: str) -> pd.DataFrame:
    """Load one AntAWS 3-hourly CSV and normalize column names/units."""
    path = Path(antaws_root) / f"{station}_3h.csv"
    if not path.exists():
        raise FileNotFoundError(f"AntAWS station file not found: {path}")
    try:
        df = pd.read_csv(path, na_values=["NA", "", "nan", "NaN"])
    except UnicodeDecodeError:
        df = pd.read_csv(path, na_values=["NA", "", "nan", "NaN"], encoding="latin1")
    required = ["Year", "Month", "Day", "Three-hourly observation time(UTC)"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    timestamp = pd.to_datetime(
        {
            "year": df["Year"].astype("Int64"),
            "month": df["Month"].astype("Int64"),
            "day": df["Day"].astype("Int64"),
            "hour": df["Three-hourly observation time(UTC)"].astype("Int64"),
        },
        errors="coerce",
        utc=True,
    )
    out = pd.DataFrame({"timestamp": timestamp})
    for raw in df.columns:
        canonical = _canonical_antaws_column(str(raw))
        if canonical is not None:
            out[canonical] = pd.to_numeric(df[raw], errors="coerce")
    out["station"] = station
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    if "air_pressure_hpa" in out.columns:
        out["air_pressure_pa"] = out["air_pressure_hpa"] * 100.0
    if "wind_direction_deg" in out.columns:
        theta = np.deg2rad(out["wind_direction_deg"].to_numpy(dtype=float))
        out["wind_dir_sin"] = np.sin(theta)
        out["wind_dir_cos"] = np.cos(theta)
    return out.reset_index(drop=True)


def load_antaws_stations(antaws_root: str | Path, stations: list[str] | tuple[str, ...]) -> pd.DataFrame:
    frames = [load_antaws_station(antaws_root, station) for station in stations]
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(["station", "timestamp"]).reset_index(drop=True)


def _regularize_station_frame(df: pd.DataFrame, *, freq_s: int) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Cannot regularize an empty station frame")
    station = str(df["station"].iloc[0])
    value_cols = [
        "air_temperature_c",
        "wind_speed_ms",
        "wind_dir_sin",
        "wind_dir_cos",
        "relative_humidity",
        "air_pressure_pa",
    ]
    frame = df.set_index("timestamp")[value_cols].sort_index()
    frame = frame.resample(f"{int(freq_s)}s").mean()
    frame = frame.interpolate(method="time", limit_direction="both")
    frame = frame.dropna(how="any")
    frame["station"] = station
    frame["timestamp"] = frame.index
    return frame.reset_index(drop=True)


def build_antaws_anchor(
    antaws_root: str | Path,
    stations: list[str] | tuple[str, ...],
    *,
    freq_s: int,
) -> pd.DataFrame:
    """Build a clean, regular AntAWS anchor series from one or more stations."""
    regularized = []
    for station in stations:
        raw = load_antaws_station(antaws_root, station)
        reg = _regularize_station_frame(raw, freq_s=freq_s)
        if len(reg) > 0:
            regularized.append(reg)
    if not regularized:
        raise ValueError(f"No usable AntAWS station data found for stations={stations}")
    return pd.concat(regularized, ignore_index=True, sort=False)


def _clean_numeric(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    series = series.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    series = series.ffill().bfill()
    return series.to_numpy(dtype=float)


def dft_phase_randomize(
    base: np.ndarray,
    *,
    steps: int,
    rng: np.random.Generator,
    keep_fraction: float = 0.15,
    shared_phase: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a DFT phase-randomized sequence while preserving amplitudes."""
    arr = _clean_numeric(base).reshape(-1)
    if arr.size < 8:
        raise ValueError("DFT synthesis needs at least 8 valid samples")
    centered = arr - float(np.mean(arr))
    coeff = np.fft.rfft(centered)
    n_freq = coeff.shape[0]
    keep = max(1, min(n_freq - 1, int(round(n_freq * float(keep_fraction)))))
    phases = np.angle(coeff)
    random_phase = shared_phase
    if random_phase is None:
        random_phase = rng.uniform(-np.pi, np.pi, size=n_freq)
    if random_phase.shape[0] != n_freq:
        raise ValueError(f"shared_phase has length {random_phase.shape[0]}, expected {n_freq}")
    phases_out = np.array(random_phase, copy=True)
    phases_out[: keep + 1] = phases[: keep + 1]
    phases_out[0] = 0.0
    if arr.size % 2 == 0:
        phases_out[-1] = 0.0
    synth = np.fft.irfft(np.abs(coeff) * np.exp(1j * phases_out), n=arr.size)
    synth = synth + float(np.mean(arr))
    if steps <= arr.size:
        return synth[:steps]
    reps = int(np.ceil(steps / arr.size))
    tiled = np.tile(synth, reps)
    return tiled[:steps]


def _match_empirical_distribution(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map values onto the empirical quantiles of a reference sequence."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    ref = np.sort(_clean_numeric(reference).reshape(-1))
    if arr.size == 0 or ref.size == 0:
        return np.array(arr, copy=True)
    ranks = stats.rankdata(arr, method="average")
    quantiles = (ranks - 0.5) / float(arr.size)
    ref_x = (np.arange(ref.size, dtype=float) + 0.5) / float(ref.size)
    return np.interp(quantiles, ref_x, ref)


def _radiation_template(timestamp: pd.Series, *, storminess: np.ndarray) -> np.ndarray:
    ts = pd.to_datetime(timestamp, utc=True)
    day_of_year = ts.dt.dayofyear.to_numpy(dtype=float)
    hour = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0
    seasonal = np.clip(np.sin(2.0 * np.pi * (day_of_year - 80.0) / 365.25), 0.0, None)
    diurnal = np.clip(np.sin(2.0 * np.pi * (hour - 6.0) / 24.0), 0.0, None)
    radiation = 420.0 * seasonal * diurnal * (1.0 - 0.35 * storminess)
    return np.maximum(0.0, radiation)


def _lowpass(values: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    out[0] = arr[0]
    for idx in range(1, arr.size):
        out[idx] = out[idx - 1] + float(alpha) * (arr[idx] - out[idx - 1])
    return out


def _clustered_event_profiles(
    *,
    steps: int,
    rng: np.random.Generator,
    target_coverage: float,
    min_duration: int,
    max_duration: int,
    lead_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(steps)
    active = np.zeros(n_steps, dtype=bool)
    precursor = np.zeros(n_steps, dtype=float)
    if n_steps <= 0 or float(target_coverage) <= 0.0:
        return active, precursor
    target_count = int(round(n_steps * float(np.clip(target_coverage, 0.0, 0.95))))
    if target_count <= 0:
        return active, precursor
    min_len = max(1, min(int(min_duration), n_steps))
    max_len = max(min_len, min(int(max_duration), n_steps))
    attempts = 0
    while int(np.sum(active)) < target_count and attempts < 1000:
        remaining = max(1, target_count - int(np.sum(active)))
        duration_hi = max(min_len, min(max_len, remaining + min_len))
        duration = int(rng.integers(min_len, duration_hi + 1))
        duration = min(duration, n_steps)
        start = int(rng.integers(0, max(1, n_steps - duration + 1)))
        active[start : start + duration] = True
        attempts += 1
    if int(np.sum(active)) < target_count:
        inactive = np.flatnonzero(~active)
        fill = inactive[: max(0, target_count - int(np.sum(active)))]
        active[fill] = True
    for start, end in _bool_runs(active):
        lead_start = max(0, int(start) - max(0, int(lead_steps)))
        if lead_start < start:
            ramp = np.linspace(0.0, 1.0, int(start) - lead_start + 1, dtype=float)[1:]
            precursor[lead_start:start] = np.maximum(precursor[lead_start:start], ramp)
    return active, precursor


def _semi_markov_event_profiles(
    *,
    steps: int,
    rng: np.random.Generator,
    target_coverage: float,
    min_duration: int,
    max_duration: int,
    min_gap: int,
    lead_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate bursty event episodes inside long storm regimes.

    The macro-regime creates sustained storm windows, while the inner event
    pulses keep individual blowing-snow durations near the Amory-style median.
    This separates "storm-heavy evaluation windows" from single event runs.
    """
    n_steps = int(steps)
    active = np.zeros(n_steps, dtype=bool)
    precursor = np.zeros(n_steps, dtype=float)
    storm_regime = np.zeros(n_steps, dtype=bool)
    if n_steps <= 0 or float(target_coverage) <= 0.0:
        return active, precursor, storm_regime

    min_len = max(1, min(int(min_duration), n_steps))
    max_len = max(min_len, min(int(max_duration), n_steps))
    min_gap_len = max(1, int(min_gap))
    within_storm_duty = 0.78
    storm_fraction = float(np.clip(float(target_coverage) / within_storm_duty, 0.08, 0.65))
    median_storm = float(np.clip(max(512.0, 18.0 * max_len), 128.0, max(128.0, n_steps / 2.0)))
    median_calm = median_storm * (1.0 - storm_fraction) / max(storm_fraction, 1e-6)
    median_calm = float(np.clip(median_calm, 128.0, max(128.0, n_steps)))

    idx = 0
    state_is_storm = bool(rng.random() < storm_fraction)
    while idx < n_steps:
        median = median_storm if state_is_storm else median_calm
        duration = int(round(rng.lognormal(mean=np.log(max(1.0, median)), sigma=0.35)))
        duration = max(24, min(duration, n_steps - idx))
        if state_is_storm:
            storm_regime[idx : idx + duration] = True
        idx += duration
        state_is_storm = not state_is_storm

    for start, end in _bool_runs(storm_regime):
        pos = int(start)
        while pos < int(end):
            if pos > start:
                pos += int(rng.integers(min_gap_len, min_gap_len + 3))
            if pos >= end:
                break
            duration = int(rng.integers(min_len, max_len + 1))
            event_end = min(int(end), pos + duration)
            active[pos:event_end] = True
            pos = event_end

    target_count = int(round(n_steps * float(np.clip(target_coverage, 0.0, 0.95))))
    if target_count > 0 and int(np.sum(active)) < int(0.75 * target_count):
        storm_indices = np.flatnonzero(storm_regime & ~active)
        fill_count = min(storm_indices.size, target_count - int(np.sum(active)))
        if fill_count > 0:
            active[storm_indices[:fill_count]] = True

    for start, end in _bool_runs(active):
        lead_start = max(0, int(start) - max(0, int(lead_steps)))
        if lead_start < start:
            ramp = np.linspace(0.0, 1.0, int(start) - lead_start + 1, dtype=float)[1:]
            precursor[lead_start:start] = np.maximum(precursor[lead_start:start], ramp)
    return active, precursor, storm_regime


def _bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(arr):
        if bool(value) and start is None:
            start = int(idx)
        elif not bool(value) and start is not None:
            runs.append((start, int(idx)))
            start = None
    if start is not None:
        runs.append((start, int(arr.size)))
    return runs


def _assign_event_subtypes(
    active: np.ndarray,
    *,
    rng: np.random.Generator,
    particle_prob: float,
    flux_prob: float,
    thermal_prob: float,
    assignment: str = "random",
    particle_eligibility: np.ndarray | None = None,
    particle_min_eligibility_fraction: float = 0.0,
) -> np.ndarray:
    """Assign one latent subtype to each event run.

    Subtype ids are intentionally simple and stored as numeric columns in the
    generated truth table:
    0 = calm/non-event, 1 = particle, 2 = flux, 3 = thermal-boundary.
    """

    arr = np.asarray(active, dtype=bool).reshape(-1)
    subtype = np.zeros(arr.shape[0], dtype=int)
    probs = np.asarray([particle_prob, flux_prob, thermal_prob], dtype=float)
    probs = np.where(np.isfinite(probs), np.maximum(probs, 0.0), 0.0)
    if float(np.sum(probs)) <= 0.0:
        probs = np.asarray([1.0, 1.0, 1.0], dtype=float)
    probs = probs / float(np.sum(probs))
    runs = _bool_runs(arr)
    eligibility = None if particle_eligibility is None else np.asarray(particle_eligibility, dtype=bool).reshape(-1)
    if eligibility is not None and eligibility.shape != arr.shape:
        raise ValueError("particle_eligibility must match active")
    min_particle_fraction = float(np.clip(particle_min_eligibility_fraction, 0.0, 1.0))
    assigned = np.zeros(3, dtype=int)
    tie_order = rng.permutation(3)
    tie_rank = np.empty(3, dtype=int)
    tie_rank[tie_order] = np.arange(3)
    for run_idx, (start, end) in enumerate(runs):
        available_types = np.ones(3, dtype=bool)
        if eligibility is not None and min_particle_fraction > 0.0:
            available_types[0] = bool(
                np.mean(eligibility[int(start) : int(end)]) >= min_particle_fraction
            )
        if str(assignment) == "stratified":
            deficit = probs * float(run_idx + 1) - assigned
            deficit = np.where(available_types, deficit, -np.inf)
            best = np.flatnonzero(np.isclose(deficit, np.max(deficit)))
            chosen = int(best[np.argmin(tie_rank[best])])
            subtype_id = chosen + 1
            assigned[chosen] += 1
        elif str(assignment) == "random":
            eligible_probs = np.where(available_types, probs, 0.0)
            eligible_probs = eligible_probs / float(np.sum(eligible_probs))
            subtype_id = int(rng.choice(np.asarray([1, 2, 3], dtype=int), p=eligible_probs))
        else:
            raise ValueError(f"Unsupported event_subtype_assignment={assignment!r}")
        subtype[int(start) : int(end)] = subtype_id
    return subtype


def _masked_lowpass_unit_noise(
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha: float,
) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(arr):
        return np.zeros(arr.shape[0], dtype=float)
    raw = rng.normal(0.0, 1.0, size=arr.shape[0])
    latent = _lowpass(raw, alpha=float(alpha))
    latent = np.where(arr, latent, 0.0)
    active = latent[arr]
    std = float(np.std(active)) if active.size else 0.0
    if std > 1.0e-6:
        latent = latent / std
    return np.where(arr, latent, 0.0)


def _lagged_effect(values: np.ndarray, *, lag_steps: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    lag = max(0, int(lag_steps))
    if lag <= 0 or arr.size == 0:
        return np.array(arr, copy=True)
    out = np.zeros_like(arr)
    if lag < arr.size:
        out[lag:] = arr[:-lag]
    return out


def _lead_context_signal(
    subtype_mask: np.ndarray,
    *,
    lead_steps: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a noisy subtype-risk context visible to the scheduler only."""

    mask = np.asarray(subtype_mask, dtype=bool).reshape(-1)
    out = mask.astype(float)
    lead = max(0, int(lead_steps))
    for start, _ in _bool_runs(mask):
        lead_start = max(0, int(start) - lead)
        if lead_start < start:
            ramp = np.linspace(0.0, 1.0, int(start) - lead_start + 1, dtype=float)[1:]
            out[lead_start:start] = np.maximum(out[lead_start:start], ramp)
    sigma = max(0.0, float(noise_std))
    if sigma > 0.0:
        out = out + rng.normal(0.0, sigma, size=out.shape[0])
    return np.clip(out, 0.0, 1.0)


def _intensity_conditioned_context_signal(
    base_signal: np.ndarray,
    target_latent: np.ndarray,
    subtype_mask: np.ndarray,
    *,
    lead_steps: int,
    strength: float,
) -> np.ndarray:
    """Blend a bounded forecast-intensity proxy into a subtype warning score."""

    base = np.asarray(base_signal, dtype=float).reshape(-1)
    latent = np.asarray(target_latent, dtype=float).reshape(-1)
    mask = np.asarray(subtype_mask, dtype=bool).reshape(-1)
    if not (base.shape == latent.shape == mask.shape):
        raise ValueError("context signal, latent, and subtype mask must have equal length")
    weight = float(np.clip(float(strength), 0.0, 1.0))
    if weight <= 0.0:
        return np.clip(base, 0.0, 1.0)

    magnitude = np.abs(latent)
    active = magnitude[mask]
    scale = float(np.quantile(active, 0.90)) if active.size else 0.0
    if not np.isfinite(scale) or scale <= 1.0e-8:
        severity = np.zeros_like(magnitude)
    else:
        severity = np.clip(magnitude / scale, 0.0, 1.0)
    forecast = np.zeros_like(severity)
    lead = max(0, int(lead_steps))
    if lead <= 0:
        forecast[:] = severity
    elif lead < severity.size:
        forecast[:-lead] = severity[lead:]
    conditioned = base * ((1.0 - weight) + weight * forecast)
    return np.clip(conditioned, 0.0, 1.0)


def _cred_probability(wind_speed: np.ndarray, *, weak_threshold: float, strong_threshold: float) -> np.ndarray:
    midpoint = 0.5 * (float(weak_threshold) + float(strong_threshold))
    scale = max(0.5, 0.25 * (float(strong_threshold) - float(weak_threshold)))
    return 1.0 / (1.0 + np.exp(-(np.asarray(wind_speed, dtype=float) - midpoint) / scale))


def _apply_event_hysteresis(
    candidate: np.ndarray,
    cred: np.ndarray,
    *,
    on_threshold: float,
    off_threshold: float,
    min_duration: int,
    min_gap: int,
) -> np.ndarray:
    candidate_arr = np.asarray(candidate, dtype=bool).reshape(-1)
    cred_arr = np.asarray(cred, dtype=float).reshape(-1)
    if candidate_arr.shape[0] != cred_arr.shape[0]:
        raise ValueError("candidate and cred must have the same length")
    out = np.zeros(candidate_arr.shape[0], dtype=bool)
    active = False
    duration = 0
    gap = max(0, int(min_gap))
    min_duration_i = max(1, int(min_duration))
    min_gap_i = max(0, int(min_gap))
    for idx, (want_event, p_event) in enumerate(zip(candidate_arr, cred_arr, strict=True)):
        if active:
            duration += 1
            can_leave = duration >= min_duration_i
            should_leave = (not bool(want_event)) or float(p_event) <= float(off_threshold)
            if can_leave and should_leave:
                active = False
                duration = 0
                gap = 1
            else:
                out[idx] = True
        else:
            can_enter = gap >= min_gap_i
            if can_enter and bool(want_event) and float(p_event) >= float(on_threshold):
                active = True
                duration = 1
                gap = 0
                out[idx] = True
            else:
                gap += 1
    return out


def generate_public_weather_truth(cfg: PublicWeatherSynthesisConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(int(cfg.seed))
    anchor = build_antaws_anchor(cfg.antaws_root, cfg.stations, freq_s=int(cfg.freq_s))
    if len(anchor) < 8:
        raise ValueError("AntAWS anchor is too short after cleaning")

    # Keep station segments contiguous. Sorting globally by timestamp would
    # interleave stations and create artificial jumps between locations.
    source = anchor.reset_index(drop=True)
    base_len = len(source)
    shared_phase = rng.uniform(-np.pi, np.pi, size=np.fft.rfft(np.zeros(base_len)).shape[0])
    synth_cols: dict[str, np.ndarray] = {}
    for col in [
        "air_temperature_c",
        "wind_speed_ms",
        "wind_dir_sin",
        "wind_dir_cos",
        "relative_humidity",
        "air_pressure_pa",
    ]:
        synth = dft_phase_randomize(
            source[col].to_numpy(dtype=float),
            steps=int(cfg.steps),
            rng=rng,
            keep_fraction=float(cfg.phase_keep_fraction),
            shared_phase=shared_phase,
        )
        if cfg.match_marginal_distribution:
            synth = _match_empirical_distribution(synth, source[col].to_numpy(dtype=float))
        synth_cols[col] = synth

    wind_speed = np.maximum(0.0, synth_cols["wind_speed_ms"])
    event_model = str(cfg.blowing_snow_event_model).strip().lower()
    storm_regime = np.zeros(int(cfg.steps), dtype=bool)
    if event_model == "clustered":
        event_candidate, event_precursor = _clustered_event_profiles(
            steps=int(cfg.steps),
            rng=rng,
            target_coverage=float(cfg.blowing_snow_event_coverage),
            min_duration=int(cfg.blowing_snow_min_duration_steps),
            max_duration=int(cfg.blowing_snow_max_duration_steps),
            lead_steps=int(cfg.blowing_snow_lead_steps),
        )
    elif event_model in {"semi_markov", "semi-markov", "v31"}:
        event_candidate, event_precursor, storm_regime = _semi_markov_event_profiles(
            steps=int(cfg.steps),
            rng=rng,
            target_coverage=float(cfg.blowing_snow_event_coverage),
            min_duration=int(cfg.blowing_snow_min_duration_steps),
            max_duration=int(cfg.blowing_snow_max_duration_steps),
            min_gap=int(cfg.blowing_snow_min_gap_steps),
            lead_steps=int(cfg.blowing_snow_lead_steps),
        )
    else:
        raise ValueError(
            f"Unsupported blowing_snow_event_model={cfg.blowing_snow_event_model!r}; "
            "expected 'clustered' or 'semi_markov'."
        )
    prelude_target = (
        float(cfg.wind_threshold_ms)
        - 2.0
        + event_precursor * (2.0 + float(cfg.blowing_snow_wind_margin_ms))
    )
    wind_speed = np.maximum(wind_speed, prelude_target)
    if np.any(storm_regime):
        storm_smooth = _lowpass(storm_regime.astype(float), alpha=0.12)
        wind_speed = wind_speed + 1.2 * storm_smooth
    event_wind_floor = (
        float(cfg.strong_wind_threshold_ms)
        + float(cfg.blowing_snow_wind_margin_ms)
        + rng.gamma(shape=1.6, scale=0.55, size=int(cfg.steps))
    )
    wind_speed = np.where(event_candidate, np.maximum(wind_speed, event_wind_floor), wind_speed)
    if event_model in {"semi_markov", "semi-markov", "v31"} and cfg.match_marginal_distribution:
        wind_speed = _match_empirical_distribution(wind_speed, source["wind_speed_ms"].to_numpy(dtype=float))
    cred = _cred_probability(
        wind_speed,
        weak_threshold=float(cfg.wind_threshold_ms),
        strong_threshold=float(cfg.strong_wind_threshold_ms),
    )
    if event_model in {"semi_markov", "semi-markov", "v31"}:
        blowing_snow_active = _apply_event_hysteresis(
            event_candidate,
            cred,
            on_threshold=float(cfg.cred_hysteresis_on),
            off_threshold=float(cfg.cred_hysteresis_off),
            min_duration=int(cfg.blowing_snow_min_duration_steps),
            min_gap=int(cfg.blowing_snow_min_gap_steps),
        )
    else:
        blowing_snow_active = event_candidate
    parsivel_available = synth_cols["air_temperature_c"] >= float(cfg.parsivel_min_temp_c)
    if bool(cfg.event_subtypes_enabled):
        event_subtype_id = _assign_event_subtypes(
            blowing_snow_active,
            rng=rng,
            particle_prob=float(cfg.event_subtype_particle_prob),
            flux_prob=float(cfg.event_subtype_flux_prob),
            thermal_prob=float(cfg.event_subtype_thermal_prob),
            assignment=str(cfg.event_subtype_assignment),
            particle_eligibility=parsivel_available,
            particle_min_eligibility_fraction=float(
                cfg.event_subtype_particle_min_parsivel_availability
            ),
        )
    else:
        event_subtype_id = np.zeros(int(cfg.steps), dtype=int)
    particle_event = event_subtype_id == 1
    flux_event = event_subtype_id == 2
    thermal_event = event_subtype_id == 3
    latent_alpha = float(cfg.event_subtype_latent_alpha)
    particle_subtype_latent = _masked_lowpass_unit_noise(
        particle_event,
        rng=rng,
        alpha=latent_alpha,
    )
    flux_subtype_latent = _masked_lowpass_unit_noise(
        flux_event,
        rng=rng,
        alpha=latent_alpha,
    )
    thermal_subtype_latent = _masked_lowpass_unit_noise(
        thermal_event,
        rng=rng,
        alpha=latent_alpha,
    )
    latent_target_lag_steps = max(0, int(cfg.event_subtype_latent_target_lag_steps))
    particle_target_latent = _lagged_effect(
        particle_subtype_latent,
        lag_steps=latent_target_lag_steps,
    )
    flux_target_latent = _lagged_effect(
        flux_subtype_latent,
        lag_steps=latent_target_lag_steps,
    )
    thermal_target_latent = _lagged_effect(
        thermal_subtype_latent,
        lag_steps=latent_target_lag_steps,
    )
    context_lead_steps = (
        int(cfg.event_subtype_context_lead_steps)
        if int(cfg.event_subtype_context_lead_steps) > 0
        else int(cfg.blowing_snow_lead_steps)
    )
    context_noise_std = max(0.0, float(cfg.event_subtype_context_noise_std))
    context_latent_strength = float(
        np.clip(float(cfg.event_subtype_context_latent_strength), 0.0, 1.0)
    )
    particle_context_alert = _lead_context_signal(
        particle_event,
        lead_steps=context_lead_steps,
        noise_std=0.0,
        rng=rng,
    )
    flux_context_alert = _lead_context_signal(
        flux_event,
        lead_steps=context_lead_steps,
        noise_std=0.0,
        rng=rng,
    )
    thermal_context_alert = _lead_context_signal(
        thermal_event,
        lead_steps=context_lead_steps,
        noise_std=0.0,
        rng=rng,
    )
    particle_context_alert = _intensity_conditioned_context_signal(
        particle_context_alert,
        particle_target_latent,
        particle_event,
        lead_steps=context_lead_steps,
        strength=context_latent_strength,
    )
    flux_context_alert = _intensity_conditioned_context_signal(
        flux_context_alert,
        flux_target_latent,
        flux_event,
        lead_steps=context_lead_steps,
        strength=context_latent_strength,
    )
    thermal_context_alert = _intensity_conditioned_context_signal(
        thermal_context_alert,
        thermal_target_latent,
        thermal_event,
        lead_steps=context_lead_steps,
        strength=context_latent_strength,
    )
    if context_noise_std > 0.0:
        particle_context_alert += rng.normal(0.0, context_noise_std, size=int(cfg.steps))
        flux_context_alert += rng.normal(0.0, context_noise_std, size=int(cfg.steps))
        thermal_context_alert += rng.normal(0.0, context_noise_std, size=int(cfg.steps))
    particle_context_alert = np.clip(particle_context_alert, 0.0, 1.0)
    flux_context_alert = np.clip(flux_context_alert, 0.0, 1.0)
    thermal_context_alert = np.clip(thermal_context_alert, 0.0, 1.0)
    if bool(cfg.event_subtypes_enabled):
        particle_smooth = _lowpass(particle_event.astype(float), alpha=0.20)
        flux_smooth = _lowpass(flux_event.astype(float), alpha=0.20)
        thermal_smooth = _lowpass(thermal_event.astype(float), alpha=0.20)
        if float(cfg.event_subtype_particle_humidity_boost_pct) != 0.0:
            synth_cols["relative_humidity"] = (
                synth_cols["relative_humidity"]
                + float(cfg.event_subtype_particle_humidity_boost_pct) * particle_smooth
            )
        if float(cfg.event_subtype_flux_wind_boost_ms) != 0.0:
            wind_speed = wind_speed + float(cfg.event_subtype_flux_wind_boost_ms) * flux_smooth
        if float(cfg.event_subtype_thermal_air_temp_drop_c) != 0.0:
            synth_cols["air_temperature_c"] = (
                synth_cols["air_temperature_c"]
                - float(cfg.event_subtype_thermal_air_temp_drop_c) * thermal_smooth
            )
    direction_rad = np.arctan2(synth_cols["wind_dir_sin"], synth_cols["wind_dir_cos"])
    wind_direction = (np.rad2deg(direction_rad) + 360.0) % 360.0
    start = pd.Timestamp(source["timestamp"].iloc[0])
    timestamp = pd.date_range(start=start, periods=int(cfg.steps), freq=f"{int(cfg.freq_s)}s", tz="UTC")

    storm_score = np.maximum(
        np.clip((wind_speed - float(cfg.strong_wind_threshold_ms)) / 6.0, 0.0, 1.0),
        np.maximum(blowing_snow_active.astype(float), event_precursor),
    )
    radiation = _radiation_template(pd.Series(timestamp), storminess=storm_score)
    surface_target = synth_cols["air_temperature_c"] - 1.1 + 0.0022 * radiation
    if bool(cfg.event_subtypes_enabled) and np.any(thermal_event):
        thermal_smooth = _lowpass(thermal_event.astype(float), alpha=0.14)
        surface_target = surface_target - float(cfg.event_subtype_thermal_surface_drop_c) * thermal_smooth
        surface_target = surface_target + (
            float(cfg.event_subtype_thermal_latent_surface_scale_c)
            * thermal_target_latent
        )
    snow_surface_temperature = _lowpass(surface_target, alpha=0.08)

    exceed = np.maximum(0.0, wind_speed - float(cfg.wind_threshold_ms))
    eps = rng.lognormal(mean=0.0, sigma=0.85, size=int(cfg.steps))
    if event_model in {"semi_markov", "semi-markov", "v31"}:
        flux_driver = np.where(blowing_snow_active, np.maximum(wind_speed, float(cfg.wind_threshold_ms)), 0.0)
        flux = 1.4e-9 * np.power(flux_driver, float(cfg.flux_wind_exponent)) * eps
    else:
        flux_exceed = np.where(blowing_snow_active, np.maximum(exceed, 0.5), exceed)
        flux = 2.2e-7 * np.power(flux_exceed, float(cfg.flux_wind_exponent)) * eps
    flux = np.where(blowing_snow_active, flux, 0.0)
    if bool(cfg.event_subtypes_enabled):
        flux = np.where(
            particle_event,
            flux * max(0.0, float(cfg.event_subtype_particle_flux_multiplier)),
            flux,
        )
        flux = np.where(
            flux_event,
            flux * max(0.0, float(cfg.event_subtype_flux_multiplier)),
            flux,
        )
        flux = np.where(
            thermal_event,
            flux * max(0.0, float(cfg.event_subtype_thermal_flux_multiplier)),
            flux,
        )
        flux = flux * np.exp(float(cfg.event_subtype_flux_latent_sigma) * flux_target_latent)
        linear_scale = max(0.0, float(cfg.event_subtype_flux_latent_linear_scale))
        if linear_scale > 0.0:
            linear_driver = np.clip(
                float(cfg.event_subtype_flux_latent_linear_offset) + flux_target_latent,
                0.0,
                max(0.0, float(cfg.event_subtype_flux_latent_linear_clip)),
            )
            flux = flux + linear_scale * flux_event.astype(float) * linear_driver
    flux = np.clip(flux, 0.0, None)

    event_microstructure = np.zeros(int(cfg.steps), dtype=float)
    particle_microstructure = np.zeros(int(cfg.steps), dtype=float)
    needs_microstructure = (
        float(cfg.event_microstructure_sigma) > 0.0
        or abs(float(cfg.event_microstructure_diameter_scale)) > 0.0
        or abs(float(cfg.event_microstructure_velocity_scale)) > 0.0
    )
    if needs_microstructure and np.any(blowing_snow_active):
        micro_raw = rng.normal(0.0, 1.0, size=int(cfg.steps))
        event_microstructure = _lowpass(micro_raw, alpha=float(cfg.event_microstructure_alpha))
        active_values = event_microstructure[blowing_snow_active]
        active_std = float(np.std(active_values)) if active_values.size else 0.0
        if active_std > 1e-6:
            event_microstructure = event_microstructure / active_std
        event_microstructure = np.where(blowing_snow_active, event_microstructure, 0.0)
        correlation = float(np.clip(float(cfg.event_particle_microstructure_correlation), -1.0, 1.0))
        if correlation >= 0.999:
            particle_microstructure = np.array(event_microstructure, copy=True)
        else:
            particle_raw = rng.normal(0.0, 1.0, size=int(cfg.steps))
            independent = _lowpass(particle_raw, alpha=float(cfg.event_microstructure_alpha))
            independent = np.where(blowing_snow_active, independent, 0.0)
            independent_values = independent[blowing_snow_active]
            independent_std = float(np.std(independent_values)) if independent_values.size else 0.0
            if independent_std > 1e-6:
                independent = independent / independent_std
            particle_microstructure = (
                correlation * event_microstructure
                + np.sqrt(max(0.0, 1.0 - correlation * correlation)) * independent
            )
            particle_values = particle_microstructure[blowing_snow_active]
            particle_std = float(np.std(particle_values)) if particle_values.size else 0.0
            if particle_std > 1e-6:
                particle_microstructure = particle_microstructure / particle_std
            particle_microstructure = np.where(blowing_snow_active, particle_microstructure, 0.0)
        flux = flux * np.exp(float(cfg.event_microstructure_sigma) * event_microstructure)
        flux = np.clip(flux, 0.0, None)

    diameter = 0.34 - 0.009 * np.minimum(wind_speed, 24.0) + rng.normal(0.0, 0.025, size=int(cfg.steps))
    diameter = diameter + float(cfg.event_microstructure_diameter_scale) * particle_microstructure
    if bool(cfg.event_subtypes_enabled):
        diameter = diameter + float(cfg.event_subtype_particle_diameter_shift_mm) * particle_event.astype(float)
        diameter = diameter + float(cfg.event_subtype_flux_diameter_shift_mm) * flux_event.astype(float)
        diameter = diameter + (
            float(cfg.event_subtype_particle_latent_diameter_scale_mm)
            * particle_target_latent
        )
    diameter = np.clip(diameter, 0.04, 0.5)
    particle_velocity = 0.35 * wind_speed + 0.8 * exceed + rng.normal(0.0, 0.25, size=int(cfg.steps))
    particle_velocity = particle_velocity + float(cfg.event_microstructure_velocity_scale) * particle_microstructure
    if bool(cfg.event_subtypes_enabled):
        particle_velocity = particle_velocity + float(cfg.event_subtype_particle_velocity_boost_ms) * particle_event.astype(float)
        particle_velocity = particle_velocity + float(cfg.event_subtype_flux_velocity_boost_ms) * flux_event.astype(float)
        particle_velocity = particle_velocity + (
            float(cfg.event_subtype_particle_latent_velocity_scale_ms)
            * particle_target_latent
        )
    particle_velocity = np.clip(particle_velocity, 0.0, 20.0)

    # Keep truth tables numeric for the existing Kalman/oracle path. The
    # availability columns carry the conditional-missingness semantics.
    diameter = np.where(blowing_snow_active & parsivel_available, diameter, 0.0)
    particle_velocity = np.where(blowing_snow_active & parsivel_available, particle_velocity, 0.0)

    df = pd.DataFrame(
        {
            "time_idx": np.arange(int(cfg.steps), dtype=int),
            "timestamp": timestamp,
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind_direction,
            "air_temperature_c": synth_cols["air_temperature_c"],
            "relative_humidity": np.clip(synth_cols["relative_humidity"], 1.0, 100.0),
            "air_pressure_pa": np.clip(synth_cols["air_pressure_pa"], 45000.0, 105000.0),
            "solar_radiation_wm2": radiation,
            "snow_surface_temperature_c": snow_surface_temperature,
            "snow_particle_mean_diameter_mm": diameter,
            "snow_particle_mean_velocity_ms": particle_velocity,
            "snow_mass_flux_kg_m2_s": flux,
            "wind_dir_sin": np.sin(np.deg2rad(wind_direction)),
            "wind_dir_cos": np.cos(np.deg2rad(wind_direction)),
            "blowing_snow_active": blowing_snow_active.astype(bool),
            "parsivel_available": parsivel_available.astype(bool),
            "event_microstructure": event_microstructure,
            "event_particle_microstructure": particle_microstructure,
            "event_subtype_id": event_subtype_id,
            "event_subtype_particle": particle_event.astype(bool),
            "event_subtype_flux": flux_event.astype(bool),
            "event_subtype_thermal": thermal_event.astype(bool),
            "event_subtype_particle_latent": particle_subtype_latent,
            "event_subtype_flux_latent": flux_subtype_latent,
            "event_subtype_thermal_latent": thermal_subtype_latent,
            "agent_context_particle_alert": particle_context_alert,
            "agent_context_flux_alert": flux_context_alert,
            "agent_context_thermal_alert": thermal_context_alert,
            "agent_context_event_alert": np.maximum.reduce(
                [particle_context_alert, flux_context_alert, thermal_context_alert]
            ),
        }
    )
    event_threshold = float(cfg.wind_threshold_ms)
    df["event_flag"] = df["blowing_snow_active"].astype(bool)
    df["storm_flag"] = df["event_flag"].astype(bool)
    meta = {
        "stations": list(cfg.stations),
        "anchor_rows": int(len(anchor)),
        "steps": int(cfg.steps),
        "freq_s": int(cfg.freq_s),
        "seed": int(cfg.seed),
        "phase_keep_fraction": float(cfg.phase_keep_fraction),
        "match_marginal_distribution": bool(cfg.match_marginal_distribution),
        "event_threshold": event_threshold,
        "blowing_snow_event_coverage_target": float(cfg.blowing_snow_event_coverage),
        "blowing_snow_event_coverage_actual": float(np.mean(df["event_flag"])),
        "blowing_snow_event_model": event_model,
        "storm_regime_fraction": float(np.mean(storm_regime)),
        "wind_ge_8_rate": float(np.mean(df["wind_speed_ms"] >= float(cfg.wind_threshold_ms))),
        "wind_ge_12_rate": float(np.mean(df["wind_speed_ms"] >= float(cfg.strong_wind_threshold_ms))),
        "event_cluster_count": int(len(_bool_runs(df["event_flag"].to_numpy(dtype=bool)))),
        "flux_wind_exponent": float(cfg.flux_wind_exponent),
        "event_microstructure_sigma": float(cfg.event_microstructure_sigma),
        "event_microstructure_alpha": float(cfg.event_microstructure_alpha),
        "event_microstructure_diameter_scale": float(cfg.event_microstructure_diameter_scale),
        "event_microstructure_velocity_scale": float(cfg.event_microstructure_velocity_scale),
        "event_particle_microstructure_correlation": float(cfg.event_particle_microstructure_correlation),
        "event_subtypes_enabled": bool(cfg.event_subtypes_enabled),
        "event_subtype_particle_rate": float(np.mean(particle_event)),
        "event_subtype_flux_rate": float(np.mean(flux_event)),
        "event_subtype_thermal_rate": float(np.mean(thermal_event)),
        "event_subtype_particle_prob": float(cfg.event_subtype_particle_prob),
        "event_subtype_assignment": str(cfg.event_subtype_assignment),
        "event_subtype_flux_prob": float(cfg.event_subtype_flux_prob),
        "event_subtype_thermal_prob": float(cfg.event_subtype_thermal_prob),
        "event_subtype_particle_flux_multiplier": float(cfg.event_subtype_particle_flux_multiplier),
        "event_subtype_flux_multiplier": float(cfg.event_subtype_flux_multiplier),
        "event_subtype_thermal_flux_multiplier": float(cfg.event_subtype_thermal_flux_multiplier),
        "event_subtype_particle_diameter_shift_mm": float(cfg.event_subtype_particle_diameter_shift_mm),
        "event_subtype_particle_velocity_boost_ms": float(cfg.event_subtype_particle_velocity_boost_ms),
        "event_subtype_flux_diameter_shift_mm": float(cfg.event_subtype_flux_diameter_shift_mm),
        "event_subtype_flux_velocity_boost_ms": float(cfg.event_subtype_flux_velocity_boost_ms),
        "event_subtype_thermal_surface_drop_c": float(cfg.event_subtype_thermal_surface_drop_c),
        "event_subtype_particle_humidity_boost_pct": float(cfg.event_subtype_particle_humidity_boost_pct),
        "event_subtype_flux_wind_boost_ms": float(cfg.event_subtype_flux_wind_boost_ms),
        "event_subtype_thermal_air_temp_drop_c": float(cfg.event_subtype_thermal_air_temp_drop_c),
        "event_subtype_latent_alpha": float(cfg.event_subtype_latent_alpha),
        "event_subtype_particle_latent_diameter_scale_mm": float(
            cfg.event_subtype_particle_latent_diameter_scale_mm
        ),
        "event_subtype_particle_min_parsivel_availability": float(
            cfg.event_subtype_particle_min_parsivel_availability
        ),
        "event_subtype_particle_latent_velocity_scale_ms": float(
            cfg.event_subtype_particle_latent_velocity_scale_ms
        ),
        "event_subtype_flux_latent_sigma": float(cfg.event_subtype_flux_latent_sigma),
        "event_subtype_thermal_latent_surface_scale_c": float(
            cfg.event_subtype_thermal_latent_surface_scale_c
        ),
        "event_subtype_latent_target_lag_steps": int(cfg.event_subtype_latent_target_lag_steps),
        "event_subtype_context_lead_steps": int(context_lead_steps),
        "event_subtype_context_noise_std": float(context_noise_std),
        "event_subtype_context_latent_strength": float(context_latent_strength),
    }
    return df, meta


def _acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    arr = _clean_numeric(values)
    arr = arr - np.mean(arr)
    denom = float(np.dot(arr, arr))
    if denom <= 0.0:
        return np.zeros(max_lag, dtype=float)
    return np.asarray([float(np.dot(arr[:-lag], arr[lag:]) / denom) for lag in range(1, max_lag + 1)])


def validate_synthetic_against_anchor(
    anchor: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    max_lag: int = 20,
) -> pd.DataFrame:
    rows = []
    pairs = [
        ("air_temperature_c", "air_temperature_c"),
        ("wind_speed_ms", "wind_speed_ms"),
        ("wind_dir_sin", "wind_dir_sin"),
        ("wind_dir_cos", "wind_dir_cos"),
        ("relative_humidity", "relative_humidity"),
        ("air_pressure_pa", "air_pressure_pa"),
    ]
    for anchor_col, synth_col in pairs:
        real = _clean_numeric(anchor[anchor_col].to_numpy(dtype=float))
        synth = _clean_numeric(synthetic[synth_col].to_numpy(dtype=float))
        n = min(len(real), len(synth))
        real_n = real[:n]
        synth_n = synth[:n]
        ks = stats.ks_2samp(real, synth)
        freq_real, psd_real = signal.welch(real_n, nperseg=min(256, n))
        freq_synth, psd_synth = signal.welch(synth_n, nperseg=min(256, n))
        del freq_real, freq_synth
        psd_mse = float(np.mean((psd_real - psd_synth) ** 2))
        acf_delta = float(np.max(np.abs(_acf(real_n, max_lag=max_lag) - _acf(synth_n, max_lag=max_lag))))
        rows.append(
            {
                "variable": synth_col,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "psd_mse": psd_mse,
                "acf_max_abs_delta_lag1_20": acf_delta,
                "real_mean": float(np.mean(real_n)),
                "synthetic_mean": float(np.mean(synth_n)),
                "real_std": float(np.std(real_n)),
                "synthetic_std": float(np.std(synth_n)),
            }
        )
    flux = synthetic["snow_mass_flux_kg_m2_s"].to_numpy(dtype=float)
    wind = synthetic["wind_speed_ms"].to_numpy(dtype=float)
    for label, mask in [
        ("flux_wind_8_12", (wind >= 8.0) & (wind < 12.0)),
        ("flux_wind_ge_12", wind >= 12.0),
    ]:
        subset = flux[mask]
        rows.append(
            {
                "variable": label,
                "ks_statistic": float("nan"),
                "ks_pvalue": float("nan"),
                "psd_mse": float("nan"),
                "acf_max_abs_delta_lag1_20": float("nan"),
                "real_mean": float("nan"),
                "synthetic_mean": float(np.mean(subset)) if subset.size else 0.0,
                "real_std": float("nan"),
                "synthetic_std": float(np.std(subset)) if subset.size else 0.0,
            }
        )
    return pd.DataFrame(rows)
