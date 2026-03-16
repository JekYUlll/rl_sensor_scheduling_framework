from __future__ import annotations


def linear_power(base_cost: float, sample_rate_scale: float = 1.0) -> float:
    return float(base_cost) * float(sample_rate_scale)


def switching_cost(prev_on: bool, next_on: bool, startup_cost: float = 0.0) -> float:
    if (not prev_on) and next_on:
        return float(startup_cost)
    return 0.0
