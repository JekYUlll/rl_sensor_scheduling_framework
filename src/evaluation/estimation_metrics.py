from __future__ import annotations

import numpy as np


def rmse_state(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    return float(np.sqrt(np.mean((x_true - x_hat) ** 2)))


def summarize_estimation(trace_p_hist: list[float], power_hist: list[float], reward_hist: list[float]) -> dict:
    return {
        "trace_P_mean": float(np.mean(trace_p_hist)) if trace_p_hist else float("nan"),
        "power_mean": float(np.mean(power_hist)) if power_hist else float("nan"),
        "reward_mean": float(np.mean(reward_hist)) if reward_hist else float("nan"),
    }
