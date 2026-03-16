from __future__ import annotations


def switch_cost(prev_action: list[str], action: list[str], weight: float) -> float:
    prev = set(prev_action)
    nxt = set(action)
    switched = len(prev.symmetric_difference(nxt))
    return float(weight) * float(switched)


def coverage_penalty(coverage_ratio: list[float], min_ratio: float) -> float:
    penalty = 0.0
    for r in coverage_ratio:
        if r < min_ratio:
            penalty += (min_ratio - r)
    return penalty
