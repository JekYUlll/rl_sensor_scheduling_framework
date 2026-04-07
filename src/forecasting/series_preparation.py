from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from forecasting.input_augmentation import augment_input_series, augment_physical_state_series


SUPPORTED_CONTEXT_SERIES_KEYS = ("trace_p", "power", "peak_power", "event_flags")


def extract_context_series(source: Mapping[str, object] | object | None) -> dict[str, np.ndarray]:
    if source is None:
        return {}
    context: dict[str, np.ndarray] = {}
    for key in SUPPORTED_CONTEXT_SERIES_KEYS:
        value = None
        if isinstance(source, Mapping):
            value = source.get(key)
        else:
            try:
                if key in source:
                    value = source[key]
            except Exception:
                value = getattr(source, key, None)
        if value is None:
            continue
        context[key] = np.asarray(value, dtype=float).reshape(-1)
    return context


def select_target_columns(
    target_series: np.ndarray,
    feature_names: list[str],
    target_columns: list[str] | None,
    *,
    time_index: np.ndarray | None = None,
    base_freq_s: int = 1,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    series_aug, names_aug, _ = augment_physical_state_series(
        np.asarray(target_series, dtype=float),
        list(feature_names),
        None,
        time_index=None if time_index is None else np.asarray(time_index, dtype=float),
        base_freq_s=int(base_freq_s),
    )
    configured = [str(name) for name in (target_columns or [])]
    if not configured:
        return np.asarray(series_aug, dtype=float), list(names_aug), None
    selected_names = [name for name in configured if name in names_aug]
    if not selected_names:
        return np.asarray(series_aug, dtype=float), list(names_aug), None
    indices = [names_aug.index(name) for name in selected_names]
    return np.asarray(series_aug[:, indices], dtype=float), selected_names, np.asarray(indices, dtype=int)


def prepare_input_and_targets(
    *,
    input_series: np.ndarray,
    target_series: np.ndarray,
    feature_names: list[str],
    observed_mask: np.ndarray | None,
    use_observed_mask: bool,
    use_time_delta: bool,
    target_columns: list[str] | None,
    time_index: np.ndarray | None = None,
    base_freq_s: int = 1,
    context_series: dict[str, np.ndarray] | None = None,
    context_features: list[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], np.ndarray | None]:
    input_aug, input_names = augment_input_series(
        input_series=np.asarray(input_series, dtype=float),
        observed_mask=None if observed_mask is None else np.asarray(observed_mask, dtype=float),
        feature_names=[str(name) for name in feature_names],
        use_observed_mask=use_observed_mask,
        use_time_delta=use_time_delta,
        time_index=None if time_index is None else np.asarray(time_index, dtype=float),
        base_freq_s=int(base_freq_s),
        context_series=None if context_series is None else extract_context_series(context_series),
        context_features=None if context_features is None else [str(name) for name in context_features],
    )
    target_sel, target_names, target_indices = select_target_columns(
        target_series=np.asarray(target_series, dtype=float),
        feature_names=[str(name) for name in feature_names],
        target_columns=target_columns,
        time_index=None if time_index is None else np.asarray(time_index, dtype=float),
        base_freq_s=int(base_freq_s),
    )
    return input_aug, input_names, target_sel, target_names, target_indices
