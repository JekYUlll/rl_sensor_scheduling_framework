from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return float("nan")
    if a.size == 1:
        return 1.0 if float(a[0]) == float(b[0]) else float("nan")
    if a.size < 3:
        # scipy.spearmanr requires at least 3 entries.
        ra = rankdata(a)
        rb = rankdata(b)
        if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])
    return float(spearmanr(a, b, nan_policy="omit").correlation)


def rank_correlation(table: pd.DataFrame, metric_col: str = "rmse") -> pd.DataFrame:
    piv = table.pivot_table(index="strategy", columns="model", values=metric_col, aggfunc="mean")
    methods = list(piv.index)
    corr = np.zeros((len(methods), len(methods)), dtype=float)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            a = piv.loc[m1].to_numpy()
            b = piv.loc[m2].to_numpy()
            corr[i, j] = _safe_spearman(a, b)
    return pd.DataFrame(corr, index=methods, columns=methods)
