from __future__ import annotations


def switch_cost(prev_action: list[str], action: list[str], weight: float) -> float:
    prev = set(prev_action)
    nxt = set(action)
    switched = len(prev.symmetric_difference(nxt))
    return float(weight) * float(switched)


def coverage_penalty(coverage_ratio: list[float], min_ratio: float) -> float:
    if not coverage_ratio or min_ratio <= 0.0:
        return 0.0
    ratios = [float(r) for r in coverage_ratio]
    shortfalls = [max(float(min_ratio) - r, 0.0) for r in ratios]
    return float(sum(shortfalls) / (len(ratios) * max(float(min_ratio), 1e-6)))
