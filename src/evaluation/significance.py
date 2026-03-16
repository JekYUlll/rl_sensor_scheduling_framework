from __future__ import annotations

import numpy as np
from scipy import stats


def paired_t_test(x: list[float], y: list[float]) -> dict:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    t, p = stats.ttest_rel(x_arr, y_arr, nan_policy="omit")
    return {"t": float(t), "p": float(p)}
