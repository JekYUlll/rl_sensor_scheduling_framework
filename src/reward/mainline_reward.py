from __future__ import annotations

from typing import Any

from scheduling.constraints import coverage_penalty


def load_training_reward_cfg(base_cfg: dict[str, Any]) -> dict[str, float]:
    reward_cfg = dict(base_cfg.get("reward", {}))
    constraints_cfg = dict(base_cfg.get("constraints", {}))
    return {
        "lambda_forecast": float(reward_cfg.get("lambda_forecast", 1.0)),
        "lambda_switch": float(reward_cfg.get("lambda_switch", 0.0)),
        "lambda_warmup_abort": float(reward_cfg.get("lambda_warmup_abort", 0.0)),
        "lambda_coverage": float(reward_cfg.get("lambda_coverage", 0.0)),
        "lambda_violation": float(reward_cfg.get("lambda_violation", 0.0)),
        "lambda_state_tracking": float(reward_cfg.get("lambda_state_tracking", 0.0)),
        "min_coverage_ratio": float(
            reward_cfg.get(
                "min_coverage_ratio",
                constraints_cfg.get("min_coverage_ratio", 0.0),
            )
        ),
    }


def instantaneous_violation_penalty(
    *,
    steady_power: float,
    peak_power: float,
    steady_limit: float,
    peak_limit: float | None,
) -> float:
    steady_violation = max(float(steady_power) - float(steady_limit), 0.0)
    steady_norm = steady_violation / max(float(steady_limit), 1e-6)
    peak_norm = 0.0
    if peak_limit is not None:
        peak_violation = max(float(peak_power) - float(peak_limit), 0.0)
        peak_norm = peak_violation / max(float(peak_limit), 1e-6)
    return float(steady_norm + peak_norm)


def compute_forecast_task_terms(
    *,
    forecast_loss: float,
    switch_count: int,
    warmup_abort_count: int = 0,
    coverage_ratio: list[float],
    steady_power: float,
    peak_power: float,
    steady_limit: float,
    peak_limit: float | None,
    reward_cfg: dict[str, float],
    state_tracking_loss: float = 0.0,
) -> dict[str, float]:
    switch_penalty = float(switch_count)
    warmup_abort_penalty = float(warmup_abort_count)
    coverage_raw = coverage_penalty(
        coverage_ratio,
        float(reward_cfg.get("min_coverage_ratio", 0.0)),
    )
    violation_raw = instantaneous_violation_penalty(
        steady_power=steady_power,
        peak_power=peak_power,
        steady_limit=steady_limit,
        peak_limit=peak_limit,
    )
    total_loss = (
        float(reward_cfg.get("lambda_forecast", 1.0)) * float(forecast_loss)
        + float(reward_cfg.get("lambda_switch", 0.0)) * switch_penalty
        + float(reward_cfg.get("lambda_warmup_abort", 0.0)) * warmup_abort_penalty
        + float(reward_cfg.get("lambda_coverage", 0.0)) * coverage_raw
        + float(reward_cfg.get("lambda_violation", 0.0)) * violation_raw
        + float(reward_cfg.get("lambda_state_tracking", 0.0)) * float(state_tracking_loss)
    )
    return {
        "forecast_loss": float(forecast_loss),
        "switch_penalty_raw": switch_penalty,
        "warmup_abort_penalty_raw": warmup_abort_penalty,
        "coverage_penalty_raw": float(coverage_raw),
        "violation_penalty_raw": float(violation_raw),
        "state_tracking_loss": float(state_tracking_loss),
        "task_loss": float(total_loss),
        "task_reward": float(-total_loss),
    }
