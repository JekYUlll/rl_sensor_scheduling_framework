from __future__ import annotations

import numpy as np


def linear_observation(c_vec: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (np.asarray(c_vec, dtype=float).reshape(1, -1) @ x.reshape(-1, 1)).reshape(-1)
