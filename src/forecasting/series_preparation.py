from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict, cast

import numpy as np

from forecasting.input_augmentation import augment_input_series, augment_physical_state_series


SUPPORTED_CONTEXT_SERIES_KEYS = ("trace_p", "power", "peak_power", "event_flags")
SUPPORTED_INPUT_FILTER_TYPES = ("causal_moving_average",)


class InputFilterConfig(TypedDict):
    type: str
    columns: list[str]
    window: int


def extract_context_series(source: Mapping[str, object] | object | None) -> dict[str, np.ndarray]:
    if source is None:
        return {}
    context: dict[str, np.ndarray] = {}
    for key in SUPPORTED_CONTEXT_SERIES_KEYS:
        value: object | None = None
        if isinstance(source, Mapping):
            value = source.get(key)
        else:
            value = getattr(source, key, None)
        if value is None:
            continue
        context[key] = np.asarray(value, dtype=float).reshape(-1)
    return context


def normalize_input_filter_cfg(input_filter_cfg: Mapping[str, object] | None) -> InputFilterConfig | None:
    if not input_filter_cfg:
        return None
    cfg = dict(input_filter_cfg)
    if not bool(cfg.get("enabled", True)):
        return None
    filter_type = str(cfg.get("type", "")).strip()
    if not filter_type:
        return None
    if filter_type not in SUPPORTED_INPUT_FILTER_TYPES:
        raise ValueError(
            f"Unsupported input filter type '{filter_type}'. "
            f"Supported types: {SUPPORTED_INPUT_FILTER_TYPES}"
        )
    raw_columns = cfg.get("columns", [])
    if isinstance(raw_columns, (str, bytes)) or not isinstance(raw_columns, Sequence):
        raise ValueError("input_filter.columns must be a sequence of feature names")
    columns = [str(name) for name in raw_columns]
    if not columns:
        return None
    if filter_type == "causal_moving_average":
        raw_window = cfg.get("window", 1)
        if not isinstance(raw_window, (int, float, str)):
            raise ValueError("input_filter.window must be an int-like value")
        window = int(raw_window)
        if window <= 1:
            return None
        return {
            "type": filter_type,
            "columns": columns,
            "window": window,
        }
    raise AssertionError(f"Unhandled input filter type: {filter_type}")


def _causal_moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if window <= 1 or arr.size == 0:
        return np.array(arr, copy=True)
    cumsum = np.cumsum(arr, dtype=float)
    out = np.empty_like(arr, dtype=float)
    for idx in range(arr.shape[0]):
        start = max(0, idx - window + 1)
        total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
        count = idx - start + 1
        out[idx] = total / float(count)
    return out


def apply_input_filter(
    input_series: np.ndarray,
    feature_names: list[str],
    input_filter_cfg: Mapping[str, object] | None,
) -> np.ndarray:
    cfg = normalize_input_filter_cfg(input_filter_cfg)
    values = np.array(input_series, dtype=float, copy=True)
    if cfg is None:
        return values
    if values.ndim != 2:
        raise ValueError(f"input_series must be 2D, got shape={values.shape}")
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    filter_type = cfg["type"]
    if filter_type == "causal_moving_average":
        window = cfg["window"]
        for column in cast(list[str], cfg["columns"]):
            idx = name_to_idx.get(str(column))
            if idx is None:
                continue
            values[:, idx] = _causal_moving_average_1d(values[:, idx], window=window)
        return values
    raise AssertionError(f"Unhandled input filter type: {filter_type}")


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
    input_filter_cfg: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], np.ndarray | None]:
    input_filtered = apply_input_filter(
        np.asarray(input_series, dtype=float),
        [str(name) for name in feature_names],
        input_filter_cfg,
    )
    input_aug, input_names = augment_input_series(
        input_series=input_filtered,
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
