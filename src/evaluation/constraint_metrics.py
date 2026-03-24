from __future__ import annotations

import numpy as np


def summarize_constraint_metrics(
    steady_power_hist: list[float],
    peak_power_hist: list[float],
    startup_extra_hist: list[float],
    average_power_budget: float | None = None,
    episode_energy_budget: float | None = None,
    peak_power_budget: float | None = None,
) -> dict[str, float]:
    steady = np.asarray(steady_power_hist, dtype=float)
    peak = np.asarray(peak_power_hist, dtype=float)
    startup_extra = np.asarray(startup_extra_hist, dtype=float)
    total_energy = float(np.sum(steady)) if steady.size else 0.0
    mean_power = float(np.mean(steady)) if steady.size else 0.0
    max_peak = float(np.max(peak)) if peak.size else 0.0
    out = {
        "power_mean": mean_power,
        "total_energy": total_energy,
        "peak_power_max": max_peak,
        "startup_extra_power_mean": float(np.mean(startup_extra)) if startup_extra.size else 0.0,
        "peak_violation_rate": 0.0,
        "avg_power_violation": 0.0,
        "energy_violation": 0.0,
    }
    if peak_power_budget is not None and peak.size:
        out["peak_violation_rate"] = float(np.mean(peak > float(peak_power_budget) + 1e-12))
    if average_power_budget is not None:
        out["avg_power_violation"] = float(mean_power - float(average_power_budget))
    if episode_energy_budget is not None:
        out["energy_violation"] = float(total_energy - float(episode_energy_budget))
    return out
