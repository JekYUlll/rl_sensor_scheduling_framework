from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v2.oracle import LinearFrozenForecastOracle, make_oracle_feature
from v2.tcn_oracle import TCNFrozenForecastOracle


def load_oracle_from_metadata(metadata: dict[str, Any], *, run_dir: str | Path, device: str = "cpu") -> Any:
    oracle_type = str(metadata.get("oracle_type", "tcn"))
    oracle_path = _resolve_path(str(metadata["oracle_path"]), Path(run_dir))
    if oracle_type == "tcn":
        return TCNFrozenForecastOracle.load(oracle_path, device=str(device))
    if oracle_type == "linear":
        return LinearFrozenForecastOracle.load(str(oracle_path))
    raise ValueError(f"Unsupported oracle_type={oracle_type!r}")


def forecast_metric_tables(
    rollout: Any,
    *,
    truth_df: pd.DataFrame,
    oracle: Any,
    metadata: dict[str, Any],
    target_columns: list[str] | tuple[str, ...] | None = None,
    target_weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    target_scales: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Evaluate a rollout by feeding its observed history through the frozen oracle."""
    state_columns = tuple(str(name) for name in rollout.state_columns)
    if not state_columns:
        raise ValueError("rollout must contain state_columns for forecast evaluation")
    missing_state = [name for name in state_columns if name not in truth_df.columns]
    if missing_state:
        raise ValueError(f"truth_df is missing rollout state columns: {missing_state}")
    targets = tuple(str(name) for name in (target_columns or metadata.get("reward_target_columns", ())))
    if not targets:
        raise ValueError("forecast evaluation requires target_columns or metadata.reward_target_columns")
    missing_targets = [name for name in targets if name not in truth_df.columns]
    if missing_targets:
        raise ValueError(f"truth_df is missing target columns: {missing_targets}")

    lookback = int(metadata.get("lookback", getattr(oracle.cfg, "lookback", 20)))
    horizon = int(metadata.get("horizon", getattr(oracle.cfg, "horizon", 1)))
    weights = _target_weights(target_weights, len(targets))
    scales = _target_scales(target_scales, len(targets))
    features, futures, current_truth = _build_forecast_windows(
        rollout,
        truth_df=truth_df,
        state_columns=state_columns,
        target_columns=targets,
        lookback=lookback,
        horizon=horizon,
        metadata=metadata,
    )
    if features.shape[0] == 0:
        return _empty_overall(rollout.policy), [], []

    predictions = oracle.predict(features).reshape(features.shape[0], horizon, len(targets))
    abs_err = np.abs(predictions - futures)
    sample_var_mae = np.nanmean(abs_err, axis=1)
    sample_weighted = _weighted_normalized_sample_mae(sample_var_mae, weights=weights, scales=scales)
    raw_sample_mae = np.nanmean(sample_var_mae, axis=1)

    condition_masks = _condition_masks(current_truth, targets=targets)
    overall: dict[str, float | int | str] = {
        "policy": str(rollout.policy),
        "forecast_samples": int(features.shape[0]),
        "forecast_weighted_mae_overall": _safe_mean(sample_weighted),
        "forecast_raw_mae_overall": _safe_mean(raw_sample_mae),
        "forecast_weighted_mae_event": _condition_mean(sample_weighted, condition_masks["event"]),
        "forecast_weighted_mae_non_event": _condition_mean(sample_weighted, condition_masks["non_event"]),
        "forecast_weighted_mae_low_temp": _condition_mean(sample_weighted, condition_masks["low_temp"]),
        "forecast_weighted_mae_normal": _condition_mean(sample_weighted, condition_masks["normal"]),
    }
    by_variable = _by_variable_rows(
        policy=str(rollout.policy),
        targets=targets,
        sample_var_mae=sample_var_mae,
        weights=weights,
        scales=scales,
    )
    by_condition = _by_condition_rows(
        policy=str(rollout.policy),
        sample_weighted=sample_weighted,
        raw_sample_mae=raw_sample_mae,
        condition_masks=condition_masks,
    )
    return overall, by_variable, by_condition


def forecast_loss_samples(
    rollout: Any,
    *,
    truth_df: pd.DataFrame,
    oracle: Any,
    metadata: dict[str, Any],
    target_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Rescore a saved rollout and return one frozen-forecaster loss per epoch.

    This is intended for evaluator-sensitivity analyses.  The rollout is not
    regenerated and the policy is not updated.  Simulator event subtypes are
    attached only after prediction so that they can be used for offline
    grouping and context-specific loss weights.
    """
    state_columns = tuple(str(name) for name in rollout.state_columns)
    if not state_columns:
        raise ValueError("rollout must contain state_columns for forecast evaluation")
    missing_state = [name for name in state_columns if name not in truth_df.columns]
    if missing_state:
        raise ValueError(f"truth_df is missing rollout state columns: {missing_state}")
    targets = tuple(str(name) for name in (target_columns or metadata.get("reward_target_columns", ())))
    if not targets:
        raise ValueError("forecast evaluation requires target_columns or metadata.reward_target_columns")
    missing_targets = [name for name in targets if name not in truth_df.columns]
    if missing_targets:
        raise ValueError(f"truth_df is missing target columns: {missing_targets}")

    lookback = int(metadata.get("lookback", getattr(oracle.cfg, "lookback", 20)))
    horizon = int(metadata.get("horizon", getattr(oracle.cfg, "horizon", 1)))
    features, futures, current_truth = _build_forecast_windows(
        rollout,
        truth_df=truth_df,
        state_columns=state_columns,
        target_columns=targets,
        lookback=lookback,
        horizon=horizon,
        metadata=metadata,
    )
    if features.shape[0] == 0:
        return pd.DataFrame(
            columns=["policy", "step_index", "event_subtype_id", "event", "forecast_loss"]
        )

    if "event_subtype_id" in current_truth.columns:
        subtype_ids = current_truth["event_subtype_id"].fillna(0).to_numpy(dtype=int)
    else:
        subtype_ids = np.zeros(features.shape[0], dtype=int)
    if "blowing_snow_event" in current_truth.columns:
        event_flags = current_truth["blowing_snow_event"].fillna(0).to_numpy(dtype=float) > 0.5
    else:
        event_flags = subtype_ids > 0

    losses = np.asarray(
        [
            oracle.loss_with_context(
                features[idx],
                futures[idx],
                context={"event_subtype_id": int(subtype_ids[idx])},
            )
            for idx in range(features.shape[0])
        ],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "policy": str(rollout.policy),
            "step_index": current_truth.index.to_numpy(dtype=int),
            "event_subtype_id": subtype_ids,
            "event": event_flags.astype(int),
            "forecast_loss": losses,
        }
    )


def _build_forecast_windows(
    rollout: Any,
    *,
    truth_df: pd.DataFrame,
    state_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    lookback: int,
    horizon: int,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    observations = np.asarray(rollout.observations, dtype=float)
    masks = np.asarray(rollout.masks, dtype=float)
    if observations.shape != masks.shape:
        raise ValueError(f"rollout observations/masks shape mismatch: {observations.shape} != {masks.shape}")
    if observations.ndim != 2:
        raise ValueError(f"rollout observations must be 2D, got {observations.shape}")

    state_mean = truth_df[list(state_columns)].to_numpy(dtype=float).mean(axis=0)
    configured_initial_mean = metadata.get("history_initial_state_mean")
    if configured_initial_mean is not None:
        configured_mean = np.asarray(configured_initial_mean, dtype=float).reshape(-1)
        if configured_mean.shape[0] != len(state_columns):
            raise ValueError("history_initial_state_mean must contain one value per state column")
        state_mean = configured_mean
    target_values = truth_df[list(target_columns)].to_numpy(dtype=float)
    segments = _infer_segments(rollout, metadata=metadata)
    features: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    current_rows: list[pd.Series] = []
    for start_idx, offset, length in segments:
        obs_seg = observations[offset : offset + length]
        mask_seg = masks[offset : offset + length]
        pad_obs = np.repeat(state_mean.reshape(1, -1), max(0, int(lookback) - 1), axis=0)
        pad_mask = np.zeros_like(pad_obs)
        obs_padded = np.vstack([pad_obs, obs_seg])
        mask_padded = np.vstack([pad_mask, mask_seg])
        for local_idx in range(int(length)):
            global_idx = int(start_idx) + int(local_idx)
            if global_idx + int(horizon) >= len(truth_df):
                continue
            window_start = int(local_idx)
            window_end = window_start + int(lookback)
            obs_window = obs_padded[window_start:window_end]
            mask_window = mask_padded[window_start:window_end]
            if obs_window.shape[0] != int(lookback):
                continue
            features.append(make_oracle_feature(obs_window, mask_window))
            futures.append(target_values[global_idx + 1 : global_idx + 1 + int(horizon)])
            current_rows.append(truth_df.iloc[global_idx])
    if not features:
        return (
            np.empty((0, int(lookback) * observations.shape[1] * 2), dtype=float),
            np.empty((0, int(horizon), len(target_columns)), dtype=float),
            pd.DataFrame(),
        )
    return np.vstack(features), np.asarray(futures, dtype=float), pd.DataFrame(current_rows)


def _infer_segments(rollout: Any, *, metadata: dict[str, Any]) -> list[tuple[int, int, int]]:
    n_steps = int(np.asarray(rollout.observations).shape[0])
    step_indices = np.asarray(getattr(rollout, "step_indices", np.asarray([], dtype=int)), dtype=int).reshape(-1)
    if step_indices.size == n_steps and n_steps > 0:
        breaks = np.flatnonzero(np.diff(step_indices) != 1) + 1
        offsets = [0, *[int(x) for x in breaks], n_steps]
        return [
            (int(step_indices[offsets[idx]]), int(offsets[idx]), int(offsets[idx + 1] - offsets[idx]))
            for idx in range(len(offsets) - 1)
            if offsets[idx + 1] > offsets[idx]
        ]
    starts = [int(x) for x in metadata.get("eval_start_indices", [0])]
    if not starts:
        starts = [0]
    eval_steps = int(metadata.get("eval_steps", max(1, n_steps // max(1, len(starts)))))
    segments: list[tuple[int, int, int]] = []
    offset = 0
    for start in starts:
        if offset >= n_steps:
            break
        length = min(eval_steps, n_steps - offset)
        segments.append((int(start), int(offset), int(length)))
        offset += length
    if offset < n_steps:
        segments.append((0, int(offset), int(n_steps - offset)))
    return segments


def _condition_masks(current_truth: pd.DataFrame, *, targets: tuple[str, ...]) -> dict[str, np.ndarray]:
    n = int(len(current_truth))
    if n == 0:
        empty = np.zeros(0, dtype=bool)
        return {"all": empty, "event": empty, "non_event": empty, "low_temp": empty, "normal": empty}
    if "wind_speed_ms" in current_truth.columns:
        event = current_truth["wind_speed_ms"].to_numpy(dtype=float) > 8.0
    else:
        event = np.zeros(n, dtype=bool)
    if "air_temperature_c" in current_truth.columns:
        low_temp = current_truth["air_temperature_c"].to_numpy(dtype=float) < -30.0
    else:
        low_temp = np.zeros(n, dtype=bool)
    all_mask = np.ones(n, dtype=bool)
    return {
        "all": all_mask,
        "event": event,
        "non_event": ~event,
        "low_temp": low_temp,
        "normal": (~event) & (~low_temp),
    }


def _by_variable_rows(
    *,
    policy: str,
    targets: tuple[str, ...],
    sample_var_mae: np.ndarray,
    weights: np.ndarray,
    scales: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for idx, target in enumerate(targets):
        raw = np.asarray(sample_var_mae[:, idx], dtype=float)
        rows.append(
            {
                "policy": policy,
                "variable": str(target),
                "forecast_samples": int(raw.size),
                "forecast_mae": _safe_mean(raw),
                "forecast_normalized_mae": _safe_mean(raw / max(float(scales[idx]), 1e-6)),
                "forecast_weight": float(weights[idx]),
                "forecast_scale": float(scales[idx]),
            }
        )
    return rows


def _by_condition_rows(
    *,
    policy: str,
    sample_weighted: np.ndarray,
    raw_sample_mae: np.ndarray,
    condition_masks: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for condition, mask in condition_masks.items():
        rows.append(
            {
                "policy": policy,
                "condition": condition,
                "forecast_samples": int(np.sum(mask)),
                "forecast_weighted_mae": _condition_mean(sample_weighted, mask),
                "forecast_raw_mae": _condition_mean(raw_sample_mae, mask),
            }
        )
    return rows


def _weighted_normalized_sample_mae(
    sample_var_mae: np.ndarray,
    *,
    weights: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    values = np.asarray(sample_var_mae, dtype=float) / np.maximum(scales.reshape(1, -1), 1e-6)
    valid = np.isfinite(values)
    weighted = np.full(values.shape[0], np.nan, dtype=float)
    for idx in range(values.shape[0]):
        row_valid = valid[idx] & np.isfinite(weights) & (weights > 0)
        if np.any(row_valid):
            weighted[idx] = float(np.average(values[idx, row_valid], weights=weights[row_valid]))
    return weighted


def _target_weights(target_weights: list[float] | tuple[float, ...] | np.ndarray | None, n_targets: int) -> np.ndarray:
    if target_weights is None:
        return np.ones(int(n_targets), dtype=float)
    weights = np.asarray(target_weights, dtype=float).reshape(-1)
    if weights.size != int(n_targets):
        raise ValueError(f"target_weights length {weights.size} does not match forecast target count {n_targets}")
    return weights


def _target_scales(target_scales: list[float] | tuple[float, ...] | np.ndarray | None, n_targets: int) -> np.ndarray:
    if target_scales is None:
        return np.ones(int(n_targets), dtype=float)
    scales = np.asarray(target_scales, dtype=float).reshape(-1)
    if scales.size != int(n_targets):
        raise ValueError(f"target_scales length {scales.size} does not match forecast target count {n_targets}")
    return np.maximum(scales, 1e-6)


def _condition_mean(values: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mask_arr = np.asarray(mask, dtype=bool)
    if arr.size == 0 or mask_arr.size != arr.size or not np.any(mask_arr):
        return float("nan")
    return _safe_mean(arr[mask_arr])


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(valid)) if valid.size else float("nan")


def _empty_overall(policy: str) -> dict[str, float | int | str]:
    return {
        "policy": str(policy),
        "forecast_samples": 0,
        "forecast_weighted_mae_overall": float("nan"),
        "forecast_raw_mae_overall": float("nan"),
        "forecast_weighted_mae_event": float("nan"),
        "forecast_weighted_mae_non_event": float("nan"),
        "forecast_weighted_mae_low_temp": float("nan"),
        "forecast_weighted_mae_normal": float("nan"),
    }


def _resolve_path(path: str, run_dir: Path) -> Path:
    raw = Path(path)
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = [Path.cwd() / raw, run_dir / raw, *[parent / raw for parent in run_dir.parents]]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw
