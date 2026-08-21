from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v2.rollout import _dtw_1d, selected_mask_diagnostics


@dataclass(frozen=True)
class LoadedRollout:
    policy: str
    observations: np.ndarray
    masks: np.ndarray
    truth: np.ndarray
    rewards: np.ndarray
    scores: np.ndarray
    powers: np.ndarray
    peaks: np.ndarray
    selected_masks: np.ndarray
    mode_ids: np.ndarray
    event_flags: np.ndarray
    oracle_losses: np.ndarray
    step_indices: np.ndarray
    warmup_abort_count: int
    sensor_ids: tuple[str, ...]
    state_columns: tuple[str, ...]


def load_rollout_npz(path: str | Path) -> LoadedRollout:
    source = Path(path)
    data = np.load(source, allow_pickle=False)
    policy = _string_scalar(data, "policy", fallback=_policy_from_path(source))
    warmup_abort_count = int(data["warmup_abort_count"][0]) if "warmup_abort_count" in data.files else -1
    return LoadedRollout(
        policy=policy,
        observations=np.asarray(data["observations"], dtype=float),
        masks=np.asarray(data["masks"], dtype=float),
        truth=np.asarray(data["truth"], dtype=float),
        rewards=np.asarray(data["rewards"], dtype=float),
        scores=np.asarray(data["scores"], dtype=float) if "scores" in data.files else np.empty((0, 0), dtype=float),
        powers=np.asarray(data["powers"], dtype=float),
        peaks=np.asarray(data["peaks"], dtype=float),
        selected_masks=np.asarray(data["selected_masks"], dtype=int),
        mode_ids=np.asarray(data["mode_ids"], dtype=int),
        event_flags=np.asarray(data["event_flags"], dtype=float),
        oracle_losses=np.asarray(data["oracle_losses"], dtype=float),
        step_indices=np.asarray(data["step_indices"], dtype=int) if "step_indices" in data.files else np.asarray([], dtype=int),
        warmup_abort_count=warmup_abort_count,
        sensor_ids=tuple(str(x) for x in data["sensor_ids"]) if "sensor_ids" in data.files else tuple(),
        state_columns=tuple(str(x) for x in data["state_columns"]) if "state_columns" in data.files else tuple(),
    )


def subset_rollout_columns(rollout: LoadedRollout, columns: list[str] | tuple[str, ...] | None) -> LoadedRollout:
    if not columns:
        return rollout
    name_to_idx = {name: idx for idx, name in enumerate(rollout.state_columns)}
    selected_names = [str(name) for name in columns if str(name) in name_to_idx]
    if not selected_names:
        raise ValueError(f"None of the requested columns are present: {columns}")
    idx = np.asarray([name_to_idx[name] for name in selected_names], dtype=int)
    return LoadedRollout(
        policy=rollout.policy,
        observations=rollout.observations[:, idx],
        masks=rollout.masks[:, idx],
        truth=rollout.truth[:, idx],
        rewards=rollout.rewards,
        scores=rollout.scores,
        powers=rollout.powers,
        peaks=rollout.peaks,
        selected_masks=rollout.selected_masks,
        mode_ids=rollout.mode_ids,
        event_flags=rollout.event_flags,
        oracle_losses=rollout.oracle_losses,
        step_indices=rollout.step_indices,
        warmup_abort_count=rollout.warmup_abort_count,
        sensor_ids=rollout.sensor_ids,
        state_columns=tuple(selected_names),
    )


def overall_metrics(
    rollout: LoadedRollout,
    *,
    per_step_budget: float | None = None,
    startup_peak_budget: float | None = None,
    dtw_window: int = 50,
    target_weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    target_scales: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> dict[str, float | int | str]:
    observed = rollout.observations
    truth = rollout.truth
    err = observed - truth
    oracle_losses = rollout.oracle_losses[np.isfinite(rollout.oracle_losses)]
    powers = rollout.powers[np.isfinite(rollout.powers)]
    peaks = rollout.peaks[np.isfinite(rollout.peaks)]
    metrics = {
        "policy": rollout.policy,
        "steps": int(observed.shape[0]),
        "weighted_normalized_mae": _weighted_normalized_mae(err, truth, target_weights, target_scales),
        "mae": _nanmean_abs(err),
        "rmse": _rmse(err),
        "smape": _smape(observed, truth),
        "pearson": _mean_pearson(observed, truth),
        "dtw": _mean_column_dtw(observed, truth, window=int(dtw_window)),
        "oracle_loss_mean": float(np.mean(oracle_losses)) if oracle_losses.size else float("nan"),
        "reward_mean": float(np.mean(rollout.rewards)) if rollout.rewards.size else float("nan"),
        "power_mean": float(np.mean(powers)) if powers.size else float("nan"),
        "power_max": float(np.max(powers)) if powers.size else float("nan"),
        "peak_power_max": float(np.max(peaks)) if peaks.size else float("nan"),
        "steady_violation_rate": _violation_rate(rollout.powers, per_step_budget),
        "peak_violation_rate": _violation_rate(rollout.peaks, startup_peak_budget),
        "event_rate": float(np.mean(rollout.event_flags)) if rollout.event_flags.size else float("nan"),
        "warmup_abort_count": int(rollout.warmup_abort_count),
        "warmup_abort_rate": float(rollout.warmup_abort_count) / max(int(observed.shape[0]), 1)
        if rollout.warmup_abort_count >= 0
        else float("nan"),
    }
    metrics.update(selected_mask_diagnostics(rollout.selected_masks))
    return metrics


def variable_metrics(
    rollout: LoadedRollout,
    *,
    dtw_window: int = 50,
    target_weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    target_scales: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> list[dict[str, float | str | int]]:
    rows = []
    weights = _target_weights(target_weights, rollout.observations.shape[1])
    scales = _target_scales(target_scales, rollout.truth)
    for idx, name in enumerate(_column_names(rollout)):
        obs = rollout.observations[:, idx]
        truth = rollout.truth[:, idx]
        err = obs - truth
        rows.append(
            {
                "policy": rollout.policy,
                "variable": name,
                "steps": int(obs.size),
                "weight": float(weights[idx]),
                "scale": float(scales[idx]),
                "normalized_mae": _normalized_mae(err, truth, scale=float(scales[idx])),
                "mae": _nanmean_abs(err),
                "rmse": _rmse(err),
                "smape": _smape(obs, truth),
                "pearson": _pearson(obs, truth),
                "dtw": _dtw_1d(obs, truth, window=int(dtw_window)) / max(int(obs.size), 1),
                "coverage": float(np.mean(rollout.masks[:, idx])) if rollout.masks.size else float("nan"),
            }
        )
    return rows


def event_group_metrics(
    rollout: LoadedRollout,
    *,
    dtw_window: int = 50,
    target_weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    target_scales: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> list[dict[str, float | str | int]]:
    flags = np.asarray(rollout.event_flags, dtype=float).reshape(-1) > 0.5
    groups = {
        "all": np.ones(rollout.observations.shape[0], dtype=bool),
        "event": flags,
        "non_event": ~flags,
    }
    rows = []
    for group, mask in groups.items():
        if not np.any(mask):
            continue
        obs = rollout.observations[mask]
        truth = rollout.truth[mask]
        err = obs - truth
        rows.append(
            {
                "policy": rollout.policy,
                "group": group,
                "steps": int(obs.shape[0]),
                "weighted_normalized_mae": _weighted_normalized_mae(err, truth, target_weights, target_scales),
                "mae": _nanmean_abs(err),
                "rmse": _rmse(err),
                "smape": _smape(obs, truth),
                "pearson": _mean_pearson(obs, truth),
                "dtw": _mean_column_dtw(obs, truth, window=int(dtw_window)),
            }
        )
    return rows


def sensor_usage_metrics(rollout: LoadedRollout) -> list[dict[str, float | str | int]]:
    if rollout.mode_ids.size == 0 or not rollout.sensor_ids:
        return []
    rows = []
    selected = rollout.selected_masks if rollout.selected_masks.size else np.zeros_like(rollout.mode_ids)
    for idx, sensor_id in enumerate(rollout.sensor_ids):
        modes = rollout.mode_ids[:, idx]
        rows.append(
            {
                "policy": rollout.policy,
                "sensor": sensor_id,
                "selected_rate": float(np.mean(selected[:, idx])) if selected.size else float("nan"),
                "off_rate": float(np.mean(modes == 0)),
                "warming_rate": float(np.mean(modes == 1)),
                "active_rate": float(np.mean(modes == 2)),
            }
        )
    return rows


def action_score_metrics(rollout: LoadedRollout) -> list[dict[str, float | str | int]]:
    if rollout.scores.size == 0 or not rollout.sensor_ids:
        return []
    selected = rollout.selected_masks if rollout.selected_masks.size else np.zeros_like(rollout.scores)
    rows = []
    for idx, sensor_id in enumerate(rollout.sensor_ids):
        scores = rollout.scores[:, idx]
        chosen = selected[:, idx].astype(bool) if selected.shape == rollout.scores.shape else np.zeros(scores.shape, dtype=bool)
        rows.append(
            {
                "policy": rollout.policy,
                "sensor": sensor_id,
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "score_min": float(np.min(scores)),
                "score_p25": float(np.quantile(scores, 0.25)),
                "score_median": float(np.quantile(scores, 0.50)),
                "score_p75": float(np.quantile(scores, 0.75)),
                "score_max": float(np.max(scores)),
                "selected_rate": float(np.mean(chosen)),
                "score_when_selected": float(np.mean(scores[chosen])) if np.any(chosen) else float("nan"),
                "score_when_not_selected": float(np.mean(scores[~chosen])) if np.any(~chosen) else float("nan"),
            }
        )
    return rows


def _policy_from_path(path: Path) -> str:
    stem = path.stem
    return stem.removeprefix("rollout_")


def _string_scalar(data: np.lib.npyio.NpzFile, key: str, *, fallback: str) -> str:
    if key not in data.files:
        return fallback
    value = data[key]
    if value.size == 0:
        return fallback
    return str(value.reshape(-1)[0])


def _column_names(rollout: LoadedRollout) -> tuple[str, ...]:
    if rollout.state_columns:
        return rollout.state_columns
    return tuple(f"var_{idx}" for idx in range(rollout.observations.shape[1]))


def _nanmean_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(np.abs(arr))) if arr.size else float("nan")


def _target_weights(
    target_weights: list[float] | tuple[float, ...] | np.ndarray | None,
    n_columns: int,
) -> np.ndarray:
    if target_weights is None:
        return np.ones(int(n_columns), dtype=float)
    weights = np.asarray(target_weights, dtype=float).reshape(-1)
    if weights.size != int(n_columns):
        raise ValueError(f"target_weights length {weights.size} does not match {n_columns} metric columns")
    return weights


def _target_scales(
    target_scales: list[float] | tuple[float, ...] | np.ndarray | None,
    truth: np.ndarray,
) -> np.ndarray:
    truth_arr = np.asarray(truth, dtype=float)
    if truth_arr.ndim != 2:
        return np.asarray([], dtype=float)
    if target_scales is not None:
        scales = np.asarray(target_scales, dtype=float).reshape(-1)
        if scales.size != truth_arr.shape[1]:
            raise ValueError(f"target_scales length {scales.size} does not match {truth_arr.shape[1]} metric columns")
        return np.maximum(scales, 1e-6)
    scales = []
    for idx in range(truth_arr.shape[1]):
        col = truth_arr[:, idx]
        valid = col[np.isfinite(col)]
        if valid.size == 0:
            scales.append(1.0)
            continue
        robust_range = float(np.nanpercentile(valid, 95) - np.nanpercentile(valid, 5))
        std = float(np.nanstd(valid))
        scales.append(max(robust_range, std, 1.0))
    return np.asarray(scales, dtype=float)


def _normalized_mae(err: np.ndarray, truth: np.ndarray, *, scale: float | None = None) -> float:
    err_arr = np.asarray(err, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    valid = np.isfinite(err_arr) & np.isfinite(truth_arr)
    if not np.any(valid):
        return float("nan")
    denom = max(float(scale) if scale is not None else float(np.nanstd(truth_arr[valid])), 1e-6)
    return float(np.nanmean(np.abs(err_arr[valid])) / denom)


def _weighted_normalized_mae(
    err: np.ndarray,
    truth: np.ndarray,
    target_weights: list[float] | tuple[float, ...] | np.ndarray | None,
    target_scales: list[float] | tuple[float, ...] | np.ndarray | None,
) -> float:
    err_arr = np.asarray(err, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    if err_arr.shape != truth_arr.shape or err_arr.ndim != 2:
        return float("nan")
    weights = _target_weights(target_weights, err_arr.shape[1])
    scales = _target_scales(target_scales, truth_arr)
    values = np.asarray(
        [
            _normalized_mae(err_arr[:, idx], truth_arr[:, idx], scale=float(scales[idx]))
            for idx in range(err_arr.shape[1])
        ],
        dtype=float,
    )
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.nanmean(arr * arr))) if arr.size else float("nan")


def _smape(pred: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(truth, dtype=float)
    denom = np.abs(p) + np.abs(t)
    valid = denom > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.nanmean(2.0 * np.abs(p[valid] - t[valid]) / denom[valid]))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _mean_pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        return float("nan")
    values = [_pearson(x[:, idx], y[:, idx]) for idx in range(x.shape[1])]
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.mean(finite)) if finite.size else float("nan")


def _mean_column_dtw(a: np.ndarray, b: np.ndarray, *, window: int) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] == 0:
        return float("nan")
    values = [_dtw_1d(x[:, idx], y[:, idx], window=int(window)) / float(x.shape[0]) for idx in range(x.shape[1])]
    return float(np.mean(values)) if values else float("nan")


def _violation_rate(values: np.ndarray, limit: float | None) -> float:
    if limit is None:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr > float(limit) + 1e-9))
