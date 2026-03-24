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


def augment_input_series(
    input_series: np.ndarray,
    observed_mask: np.ndarray | None,
    feature_names: list[str],
    *,
    use_observed_mask: bool,
    use_time_delta: bool,
) -> tuple[np.ndarray, list[str]]:
    series = np.asarray(input_series, dtype=float)
    names = list(feature_names)
    parts = [series]

    if use_observed_mask or use_time_delta:
        if observed_mask is None:
            raise ValueError("observed_mask is required for missing-aware input augmentation")
        mask = np.asarray(observed_mask, dtype=float)
        if mask.shape != series.shape:
            raise ValueError(f"observed_mask shape {mask.shape} does not match input_series shape {series.shape}")
        if use_observed_mask:
            parts.append(mask)
            names.extend([f"is_observed_{name}" for name in feature_names])
        if use_time_delta:
            delta = compute_time_since_observed(mask)
            parts.append(delta)
            names.extend([f"delta_{name}" for name in feature_names])

    return np.concatenate(parts, axis=1), names
