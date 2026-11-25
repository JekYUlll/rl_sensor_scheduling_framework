from __future__ import annotations

import numpy as np


def fit_linear_gaussian_dynamics(
    values: np.ndarray,
    ridge_lambda: float = 1e-4,
    fit_intercept: bool = True,
    max_spectral_radius: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2:
        raise ValueError("values must be a 2D array")
    n_steps, state_dim = data.shape
    if n_steps < 3:
        return (
            np.eye(state_dim, dtype=float),
            np.zeros(state_dim, dtype=float),
            np.eye(state_dim, dtype=float) * 1e-3,
        )

    x_prev = data[:-1]
    x_next = data[1:]
    if fit_intercept:
        design = np.concatenate([x_prev, np.ones((x_prev.shape[0], 1), dtype=float)], axis=1)
    else:
        design = x_prev

    gram = design.T @ design
    reg = np.eye(gram.shape[0], dtype=float) * float(ridge_lambda)
    if fit_intercept:
        reg[-1, -1] = 0.0
    coef = np.linalg.solve(gram + reg, design.T @ x_next)

    if fit_intercept:
        a_mat = coef[:-1].T
        b_vec = coef[-1].astype(float)
    else:
        a_mat = coef.T
        b_vec = np.zeros(state_dim, dtype=float)

    eigvals = np.linalg.eigvals(a_mat)
    spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    if spectral_radius > max_spectral_radius and spectral_radius > 1e-12:
        a_mat = a_mat * (max_spectral_radius / spectral_radius)

    residual = x_next - (x_prev @ a_mat.T + b_vec)
    if residual.shape[0] >= 2:
        q_mat = np.cov(residual, rowvar=False)
        if np.ndim(q_mat) == 0:
            q_mat = np.array([[float(q_mat)]], dtype=float)
    else:
        q_mat = np.eye(state_dim, dtype=float) * 1e-3
    q_mat = np.asarray(q_mat, dtype=float)
    q_mat += 1e-6 * np.eye(state_dim, dtype=float)
    return a_mat, b_vec, q_mat


def safe_feature_scale(values: np.ndarray, min_scale: float = 1e-6) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2:
        raise ValueError("values must be a 2D array")
    scale = data.std(axis=0, ddof=1)
    return np.maximum(scale, float(min_scale))


def target_relevance_weights(
    values: np.ndarray,
    state_columns: list[str],
    target_columns: list[str] | None,
    min_weight: float = 0.25,
    power: float = 1.0,
) -> np.ndarray:
    n_features = len(state_columns)
    if not target_columns:
        return np.ones(n_features, dtype=float)

    data = np.asarray(values, dtype=float)
    col_to_idx = {name: i for i, name in enumerate(state_columns)}
    target_indices = [col_to_idx[name] for name in target_columns if name in col_to_idx]
    if not target_indices:
        return np.ones(n_features, dtype=float)

    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.zeros(n_features, dtype=float)
    for idx in range(n_features):
        raw[idx] = float(max(abs(corr[idx, tid]) for tid in target_indices))

    raw = np.power(np.clip(raw, 0.0, None), float(power))
    max_raw = float(raw.max())
    if max_raw <= 1e-12:
        return np.ones(n_features, dtype=float)

    scaled = float(min_weight) + (1.0 - float(min_weight)) * (raw / max_raw)
    return scaled / np.mean(scaled)
