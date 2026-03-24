from __future__ import annotations

import numpy as np


def pearson_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def smape_1d(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(b - a) / denom) * 100.0)


def dtw_distance_1d(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int | None = None,
    normalize: bool = True,
) -> float:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = a.size
    m = b.size
    if window is None:
        window = max(5, abs(n - m), int(0.1 * max(n, m)))
    else:
        window = max(int(window), abs(n - m))

    inf = float("inf")
    prev = np.full(m + 1, inf, dtype=float)
    curr = np.full(m + 1, inf, dtype=float)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr.fill(inf)
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = abs(a[i - 1] - b[j - 1])
            curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
        prev, curr = curr, prev

    dist = float(prev[m])
    if normalize:
        dist /= float(max(n, m))
    return dist

