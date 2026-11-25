from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OracleConfig:
    lookback: int = 20
    horizon: int = 5
    ridge_alpha: float = 1.0
    target_weights: tuple[float, ...] | None = None
    target_scales: tuple[float, ...] | None = None
    normalized_loss: bool = True
    subtype_loss_weighting: bool = False
    subtype_particle_target_weights: tuple[float, ...] | None = None
    subtype_flux_target_weights: tuple[float, ...] | None = None
    subtype_thermal_target_weights: tuple[float, ...] | None = None


class LinearFrozenForecastOracle:
    """Small frozen multi-output ridge oracle for v2 smoke/full-chain runs.

    The model is intentionally lightweight: it provides a stable forecast-loss
    reward without pulling the v2 pipeline into the historical torch stack.
    """

    def __init__(self, cfg: OracleConfig) -> None:
        self.cfg = cfg
        self.coef_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: np.ndarray | None = None
        self.y_std_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.n_targets_: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self.coef_ is not None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        sample_contexts: np.ndarray | list[dict[str, object] | None] | None = None,
    ) -> "LinearFrozenForecastOracle":
        del sample_contexts
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 2 or y_arr.ndim != 2:
            raise ValueError(f"x and y must be 2D, got {x_arr.shape=} {y_arr.shape=}")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"x/y sample mismatch: {x_arr.shape[0]} != {y_arr.shape[0]}")
        self.x_mean_ = x_arr.mean(axis=0)
        self.x_std_ = np.maximum(x_arr.std(axis=0), 1e-6)
        self.y_mean_ = y_arr.mean(axis=0)
        self.y_std_ = np.maximum(y_arr.std(axis=0), 1e-6)
        x_norm = (x_arr - self.x_mean_) / self.x_std_
        y_norm = (y_arr - self.y_mean_) / self.y_std_
        design = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=float)], axis=1)
        reg = float(self.cfg.ridge_alpha) * np.eye(design.shape[1], dtype=float)
        reg[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + reg, design.T @ y_norm)
        self.n_features_ = int(x_arr.shape[1])
        self.n_targets_ = int(y_arr.shape[1])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.x_mean_ is None or self.x_std_ is None or self.y_mean_ is None or self.y_std_ is None:
            raise RuntimeError("Oracle is not fitted")
        x_arr = np.asarray(x, dtype=float)
        was_1d = x_arr.ndim == 1
        if was_1d:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2:
            raise ValueError(f"x must be 1D or 2D, got {x_arr.shape}")
        x_norm = (x_arr - self.x_mean_) / self.x_std_
        design = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=float)], axis=1)
        y_norm = design @ self.coef_
        pred = y_norm * self.y_std_ + self.y_mean_
        return pred[0] if was_1d else pred

    def loss(self, feature: np.ndarray, future: np.ndarray) -> float:
        return self.loss_with_context(feature, future, context=None)

    def loss_with_context(self, feature: np.ndarray, future: np.ndarray, *, context: dict[str, object] | None = None) -> float:
        pred_flat = self.predict(feature).reshape(-1)
        true_flat = np.asarray(future, dtype=float).reshape(-1)
        if pred_flat.shape != true_flat.shape:
            raise ValueError(f"Prediction/target shape mismatch: {pred_flat.shape} != {true_flat.shape}")
        errors = np.abs(pred_flat - true_flat)
        if self.cfg.normalized_loss:
            scale = self._flat_scales(len(errors))
            errors = errors / scale
        weights = self._flat_weights(len(errors), context=context)
        return float(np.sum(errors * weights) / np.sum(weights))

    def _flat_weights(self, n: int, *, context: dict[str, object] | None = None) -> np.ndarray:
        target_weights = self._target_weights_for_context(context)
        if target_weights is None:
            return np.ones(n, dtype=float)
        base = np.asarray(target_weights, dtype=float).reshape(-1)
        if base.size == 0:
            return np.ones(n, dtype=float)
        reps = int(np.ceil(n / base.size))
        return np.tile(base, reps)[:n]

    def _target_weights_for_context(self, context: dict[str, object] | None) -> tuple[float, ...] | None:
        if not bool(self.cfg.subtype_loss_weighting):
            return self.cfg.target_weights
        subtype_id = 0
        if context is not None:
            subtype_id = int(context.get("event_subtype_id", 0) or 0)
        if subtype_id == 1 and self.cfg.subtype_particle_target_weights is not None:
            return self.cfg.subtype_particle_target_weights
        if subtype_id == 2 and self.cfg.subtype_flux_target_weights is not None:
            return self.cfg.subtype_flux_target_weights
        if subtype_id == 3 and self.cfg.subtype_thermal_target_weights is not None:
            return self.cfg.subtype_thermal_target_weights
        return self.cfg.target_weights

    def _flat_scales(self, n: int) -> np.ndarray:
        if self.cfg.target_scales is not None:
            base = np.asarray(self.cfg.target_scales, dtype=float).reshape(-1)
        elif self.y_std_ is not None:
            base = np.asarray(self.y_std_, dtype=float).reshape(-1)
        else:
            base = np.ones(n, dtype=float)
        if base.size == 0:
            return np.ones(n, dtype=float)
        reps = int(np.ceil(n / base.size))
        return np.maximum(np.tile(base, reps)[:n], 1e-6)

    def save(self, path: str) -> None:
        if self.coef_ is None or self.x_mean_ is None or self.x_std_ is None or self.y_mean_ is None or self.y_std_ is None:
            raise RuntimeError("Cannot save an unfitted oracle")
        np.savez(
            path,
            coef=self.coef_,
            x_mean=self.x_mean_,
            x_std=self.x_std_,
            y_mean=self.y_mean_,
            y_std=self.y_std_,
            lookback=np.asarray([self.cfg.lookback], dtype=int),
            horizon=np.asarray([self.cfg.horizon], dtype=int),
            ridge_alpha=np.asarray([self.cfg.ridge_alpha], dtype=float),
            normalized_loss=np.asarray([int(self.cfg.normalized_loss)], dtype=int),
            target_weights=np.asarray([] if self.cfg.target_weights is None else self.cfg.target_weights, dtype=float),
            target_scales=np.asarray([] if self.cfg.target_scales is None else self.cfg.target_scales, dtype=float),
            subtype_loss_weighting=np.asarray([int(self.cfg.subtype_loss_weighting)], dtype=int),
            subtype_particle_target_weights=np.asarray(
                [] if self.cfg.subtype_particle_target_weights is None else self.cfg.subtype_particle_target_weights,
                dtype=float,
            ),
            subtype_flux_target_weights=np.asarray(
                [] if self.cfg.subtype_flux_target_weights is None else self.cfg.subtype_flux_target_weights,
                dtype=float,
            ),
            subtype_thermal_target_weights=np.asarray(
                [] if self.cfg.subtype_thermal_target_weights is None else self.cfg.subtype_thermal_target_weights,
                dtype=float,
            ),
        )

    @classmethod
    def load(cls, path: str) -> "LinearFrozenForecastOracle":
        data = np.load(path, allow_pickle=False)
        weights = data["target_weights"]
        scales = data["target_scales"] if "target_scales" in data.files else np.asarray([], dtype=float)
        particle_weights = data["subtype_particle_target_weights"] if "subtype_particle_target_weights" in data.files else np.asarray([], dtype=float)
        flux_weights = data["subtype_flux_target_weights"] if "subtype_flux_target_weights" in data.files else np.asarray([], dtype=float)
        thermal_weights = data["subtype_thermal_target_weights"] if "subtype_thermal_target_weights" in data.files else np.asarray([], dtype=float)
        cfg = OracleConfig(
            lookback=int(data["lookback"][0]),
            horizon=int(data["horizon"][0]),
            ridge_alpha=float(data["ridge_alpha"][0]),
            normalized_loss=bool(int(data["normalized_loss"][0])),
            target_weights=None if weights.size == 0 else tuple(float(x) for x in weights),
            target_scales=None if scales.size == 0 else tuple(float(x) for x in scales),
            subtype_loss_weighting=bool(int(data["subtype_loss_weighting"][0])) if "subtype_loss_weighting" in data.files else False,
            subtype_particle_target_weights=None if particle_weights.size == 0 else tuple(float(x) for x in particle_weights),
            subtype_flux_target_weights=None if flux_weights.size == 0 else tuple(float(x) for x in flux_weights),
            subtype_thermal_target_weights=None if thermal_weights.size == 0 else tuple(float(x) for x in thermal_weights),
        )
        oracle = cls(cfg)
        oracle.coef_ = np.asarray(data["coef"], dtype=float)
        oracle.x_mean_ = np.asarray(data["x_mean"], dtype=float)
        oracle.x_std_ = np.asarray(data["x_std"], dtype=float)
        oracle.y_mean_ = np.asarray(data["y_mean"], dtype=float)
        oracle.y_std_ = np.asarray(data["y_std"], dtype=float)
        oracle.n_features_ = int(oracle.x_mean_.shape[0])
        oracle.n_targets_ = int(oracle.y_mean_.shape[0])
        return oracle


def make_oracle_feature(history: np.ndarray, mask_history: np.ndarray) -> np.ndarray:
    h = np.asarray(history, dtype=float)
    m = np.asarray(mask_history, dtype=float)
    if h.shape != m.shape:
        raise ValueError(f"history/mask shape mismatch: {h.shape} != {m.shape}")
    return np.concatenate([h.reshape(-1), m.reshape(-1)], axis=0)


def build_supervised_windows(
    observed_series: np.ndarray,
    mask_series: np.ndarray,
    target_series: np.ndarray,
    *,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed_series, dtype=float)
    mask = np.asarray(mask_series, dtype=float)
    target = np.asarray(target_series, dtype=float)
    if obs.shape != mask.shape:
        raise ValueError(f"observed/mask shape mismatch: {obs.shape} != {mask.shape}")
    if obs.shape[0] != target.shape[0]:
        raise ValueError(f"observed/target length mismatch: {obs.shape[0]} != {target.shape[0]}")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for end in range(int(lookback), obs.shape[0] - int(horizon)):
        xs.append(make_oracle_feature(obs[end - int(lookback) : end], mask[end - int(lookback) : end]))
        ys.append(target[end : end + int(horizon)].reshape(-1))
    if not xs:
        raise ValueError("Not enough samples to build oracle windows")
    return np.vstack(xs), np.vstack(ys)


def build_supervised_windows_with_context(
    observed_series: np.ndarray,
    mask_series: np.ndarray,
    target_series: np.ndarray,
    *,
    context_series: np.ndarray,
    lookback: int,
    horizon: int,
    context_offset: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(observed_series, dtype=float)
    mask = np.asarray(mask_series, dtype=float)
    target = np.asarray(target_series, dtype=float)
    context = np.asarray(context_series).reshape(-1)
    if obs.shape != mask.shape:
        raise ValueError(f"observed/mask shape mismatch: {obs.shape} != {mask.shape}")
    if obs.shape[0] != target.shape[0]:
        raise ValueError(f"observed/target length mismatch: {obs.shape[0]} != {target.shape[0]}")
    if obs.shape[0] != context.shape[0]:
        raise ValueError(f"observed/context length mismatch: {obs.shape[0]} != {context.shape[0]}")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    contexts: list[object] = []
    for end in range(int(lookback), obs.shape[0] - int(horizon)):
        context_idx = int(end) + int(context_offset)
        if context_idx < 0 or context_idx >= context.shape[0]:
            raise ValueError(
                f"context_offset={context_offset} gives out-of-range index {context_idx} "
                f"for window ending at {end}"
            )
        xs.append(make_oracle_feature(obs[end - int(lookback) : end], mask[end - int(lookback) : end]))
        ys.append(target[end : end + int(horizon)].reshape(-1))
        contexts.append(context[context_idx])
    if not xs:
        raise ValueError("Not enough samples to build oracle windows")
    return np.vstack(xs), np.vstack(ys), np.asarray(contexts)
