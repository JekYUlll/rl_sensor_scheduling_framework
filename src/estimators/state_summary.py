from __future__ import annotations

from typing import Any


def flatten_rl_state(rl_state: dict[str, Any]) -> list[float]:
    values: list[float] = []
    values.extend(float(v) for v in rl_state.get("x_hat", []))
    values.extend(float(v) for v in rl_state.get("diag_P", []))
    values.append(float(rl_state.get("trace_P", 0.0)))
    values.extend(float(v) for v in rl_state.get("freshness", []))
    values.extend(float(v) for v in rl_state.get("coverage_ratio", []))
    values.append(float(rl_state.get("budget_ratio", 1.0)))
    values.extend(float(v) for v in rl_state.get("previous_action", []))
    return values
