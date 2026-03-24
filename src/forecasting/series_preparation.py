from __future__ import annotations

import numpy as np

from forecasting.input_augmentation import augment_input_series, augment_physical_state_series


def select_target_columns(
    target_series: np.ndarray,
    feature_names: list[str],
    target_columns: list[str] | None,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    series_aug, names_aug, _ = augment_physical_state_series(np.asarray(target_series, dtype=float), list(feature_names), None)
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
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], np.ndarray | None]:
    input_aug, input_names = augment_input_series(
        input_series=np.asarray(input_series, dtype=float),
        observed_mask=None if observed_mask is None else np.asarray(observed_mask, dtype=float),
        feature_names=[str(name) for name in feature_names],
        use_observed_mask=use_observed_mask,
        use_time_delta=use_time_delta,
    )
    target_sel, target_names, target_indices = select_target_columns(
        target_series=np.asarray(target_series, dtype=float),
        feature_names=[str(name) for name in feature_names],
        target_columns=target_columns,
    )
    return input_aug, input_names, target_sel, target_names, target_indices
