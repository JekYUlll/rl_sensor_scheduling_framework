from __future__ import annotations

from typing import Any


def flatten_rl_state(rl_state: dict[str, Any]) -> list[float]:
    values: list[float] = []
    x_hat = rl_state.get("x_hat_scaled", rl_state.get("x_hat", []))
    diag_p = rl_state.get("diag_P_norm", rl_state.get("diag_P", []))
    trace_p = rl_state.get("weighted_trace_P_norm", rl_state.get("trace_P_norm", rl_state.get("trace_P", 0.0)))
    values.extend(float(v) for v in x_hat)
    values.extend(float(v) for v in diag_p)
    values.append(float(trace_p))
    values.extend(float(v) for v in rl_state.get("freshness", []))
    values.extend(float(v) for v in rl_state.get("coverage_ratio", []))
    values.append(float(rl_state.get("budget_ratio", 1.0)))
    values.extend(float(v) for v in rl_state.get("previous_action", []))
    values.extend(float(v) for v in rl_state.get("warming_mask", []))
    values.extend(float(v) for v in rl_state.get("ready_mask", []))
    values.extend(float(v) for v in rl_state.get("warm_remaining_norm", []))
    values.append(1.0 if bool(rl_state.get("event", False)) else 0.0)
    if "time_of_day_sin" in rl_state:
        values.append(float(rl_state["time_of_day_sin"]))
    if "time_of_day_cos" in rl_state:
        values.append(float(rl_state["time_of_day_cos"]))
    return values
