from __future__ import annotations

import numpy as np


def compute_time_since_observed(observed_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(observed_mask, dtype=float)
    if mask.ndim != 2:
        raise ValueError(f"observed_mask must be 2D, got shape={mask.shape}")
    delta = np.zeros_like(mask, dtype=float)
    for t in range(1, mask.shape[0]):
        delta[t] = (delta[t - 1] + 1.0) * (1.0 - mask[t - 1])
        delta[t, mask[t] > 0.5] = 0.0
    return delta


def _safe_col(series: np.ndarray, names: list[str], name: str) -> np.ndarray | None:
    if name not in names:
        return None
    return series[:, names.index(name)]


def _safe_mask(mask: np.ndarray | None, names: list[str], name: str) -> np.ndarray | None:
    if mask is None or name not in names:
        return None
    return mask[:, names.index(name)]


def _append_feature(
    parts: list[np.ndarray],
    names_out: list[str],
    masks_out: list[np.ndarray] | None,
    values: np.ndarray,
    name: str,
    mask_values: np.ndarray | None,
) -> None:
    parts.append(np.asarray(values, dtype=float).reshape(-1, 1))
    names_out.append(name)
    if masks_out is not None:
        if mask_values is None:
            mask_values = np.ones_like(values, dtype=float)
        masks_out.append(np.asarray(mask_values, dtype=float).reshape(-1, 1))


def augment_physical_state_series(
    series: np.ndarray,
    feature_names: list[str],
    observed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"series must be 2D, got shape={values.shape}")
    mask = None if observed_mask is None else np.asarray(observed_mask, dtype=float)
    if mask is not None and mask.shape != values.shape:
        raise ValueError(f"observed_mask shape {mask.shape} does not match series shape {values.shape}")

    parts: list[np.ndarray] = [values]
    names_out = [str(name) for name in feature_names]
    masks_out: list[np.ndarray] | None = None if mask is None else [mask]

    wind_speed = _safe_col(values, names_out, "wind_speed_ms")
    wind_dir = _safe_col(values, names_out, "wind_direction_deg")
    air_temp = _safe_col(values, names_out, "air_temperature_c")
    surface_temp = _safe_col(values, names_out, "snow_surface_temperature_c")
    particle_velocity = _safe_col(values, names_out, "snow_particle_mean_velocity_ms")
    particle_diameter = _safe_col(values, names_out, "snow_particle_mean_diameter_mm")

    wind_speed_mask = _safe_mask(mask, names_out, "wind_speed_ms")
    wind_dir_mask = _safe_mask(mask, names_out, "wind_direction_deg")
    air_temp_mask = _safe_mask(mask, names_out, "air_temperature_c")
    surface_temp_mask = _safe_mask(mask, names_out, "snow_surface_temperature_c")
    particle_velocity_mask = _safe_mask(mask, names_out, "snow_particle_mean_velocity_ms")
    particle_diameter_mask = _safe_mask(mask, names_out, "snow_particle_mean_diameter_mm")

    if wind_dir is not None:
        theta = np.deg2rad(wind_dir)
        _append_feature(parts, names_out, masks_out, np.sin(theta), "wind_dir_sin", wind_dir_mask)
        _append_feature(parts, names_out, masks_out, np.cos(theta), "wind_dir_cos", wind_dir_mask)

    if wind_speed is not None and wind_dir is not None:
        theta = np.deg2rad(wind_dir)
        combined_mask = None
        if wind_speed_mask is not None and wind_dir_mask is not None:
            combined_mask = wind_speed_mask * wind_dir_mask
        _append_feature(parts, names_out, masks_out, wind_speed * np.cos(theta), "wind_u", combined_mask)
        _append_feature(parts, names_out, masks_out, wind_speed * np.sin(theta), "wind_v", combined_mask)

    if surface_temp is not None and air_temp is not None:
        combined_mask = None
        if surface_temp_mask is not None and air_temp_mask is not None:
            combined_mask = surface_temp_mask * air_temp_mask
        _append_feature(parts, names_out, masks_out, surface_temp - air_temp, "surface_air_temp_gap", combined_mask)

    if particle_velocity is not None:
        _append_feature(
            parts,
            names_out,
            masks_out,
            particle_velocity**2,
            "particle_kinetic_proxy",
            particle_velocity_mask,
        )

    if particle_velocity is not None and particle_diameter is not None:
        combined_mask = None
        if particle_velocity_mask is not None and particle_diameter_mask is not None:
            combined_mask = particle_velocity_mask * particle_diameter_mask
        _append_feature(
            parts,
            names_out,
            masks_out,
            particle_velocity * particle_diameter,
            "size_velocity_interaction",
            combined_mask,
        )

    if wind_speed is not None and air_temp is not None:
        threshold = np.maximum(6.5, 0.45 * np.sqrt(np.abs(air_temp)) + 5.5)
        combined_mask = None
        if wind_speed_mask is not None and air_temp_mask is not None:
            combined_mask = wind_speed_mask * air_temp_mask
        _append_feature(
            parts,
            names_out,
            masks_out,
            np.maximum(wind_speed - threshold, 0.0),
            "transport_exceedance",
            combined_mask,
        )

    series_out = np.concatenate(parts, axis=1)
    mask_out = None if masks_out is None else np.concatenate(masks_out, axis=1)
    return series_out, names_out, mask_out


def augment_input_series(
    input_series: np.ndarray,
    observed_mask: np.ndarray | None,
    feature_names: list[str],
    *,
    use_observed_mask: bool,
    use_time_delta: bool,
) -> tuple[np.ndarray, list[str]]:
    series, names, mask = augment_physical_state_series(
        np.asarray(input_series, dtype=float),
        list(feature_names),
        None if observed_mask is None else np.asarray(observed_mask, dtype=float),
    )
    parts = [series]

    if use_observed_mask or use_time_delta:
        if mask is None:
            raise ValueError("observed_mask is required for missing-aware input augmentation")
        if use_observed_mask:
            parts.append(mask)
            names.extend([f"is_observed_{name}" for name in names[: mask.shape[1]]])
        if use_time_delta:
            delta = compute_time_since_observed(mask)
            parts.append(delta)
            names.extend([f"delta_{name}" for name in names[: delta.shape[1]]])

    return np.concatenate(parts, axis=1), names
