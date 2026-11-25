from __future__ import annotations

import numpy as np


def split_event_non_event(y_true: np.ndarray, y_pred: np.ndarray, event_flags: np.ndarray) -> dict:
    event_flags = np.asarray(event_flags).astype(bool)
    err = (np.asarray(y_pred) - np.asarray(y_true)) ** 2

    def _rmse(mask):
        if not np.any(mask):
            return float("nan")
        return float(np.sqrt(np.mean(err[mask])))

    return {
        "event_rmse": _rmse(event_flags),
        "non_event_rmse": _rmse(~event_flags),
    }
