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


def _resample_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size <= target_len:
        return arr
    src_x = np.arange(arr.size, dtype=float)
    dst_x = np.linspace(0.0, float(arr.size - 1), num=target_len, dtype=float)
    if np.isfinite(arr).all():
        return np.interp(dst_x, src_x, arr).astype(float, copy=False)
    dst_idx = np.clip(np.rint(dst_x).astype(int), 0, arr.size - 1)
    return arr[dst_idx]


def dtw_distance_1d(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int | None = None,
    normalize: bool = True,
    max_points: int | None = 1024,
) -> float:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    orig_n = a.size
    orig_m = b.size
    if max_points is not None and max_points > 1:
        a = _resample_1d(a, int(max_points))
        b = _resample_1d(b, int(max_points))
    n = a.size
    m = b.size
    if window is None:
        window = max(5, abs(n - m), int(0.1 * max(n, m)))
    else:
        if max_points is not None and max_points > 1 and (orig_n != n or orig_m != m):
            scale = max(float(n) / max(float(orig_n), 1.0), float(m) / max(float(orig_m), 1.0))
            window = max(1, int(round(float(window) * scale)))
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
