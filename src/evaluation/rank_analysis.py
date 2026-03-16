from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_correlation(table: pd.DataFrame, metric_col: str = "rmse") -> pd.DataFrame:
    piv = table.pivot_table(index="strategy", columns="model", values=metric_col, aggfunc="mean")
    methods = list(piv.index)
    corr = np.zeros((len(methods), len(methods)), dtype=float)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            a = piv.loc[m1].to_numpy()
            b = piv.loc[m2].to_numpy()
            corr[i, j] = spearmanr(a, b, nan_policy="omit").correlation
    return pd.DataFrame(corr, index=methods, columns=methods)
