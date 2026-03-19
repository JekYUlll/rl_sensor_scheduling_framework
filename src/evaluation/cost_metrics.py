from __future__ import annotations

from scheduling.constraints import coverage_penalty


def select_uncertainty_value(uncertainty_summary: dict, cost_cfg: dict) -> float:
    metric = str(cost_cfg.get("uncertainty_metric", "trace_P"))
    value = uncertainty_summary.get(metric)
    if value is None:
        value = uncertainty_summary.get("trace_P", 0.0)
    return float(value)


def compute_step_cost(
    uncertainty_summary: dict,
    power_cost: float,
    switch_count: int,
    coverage_ratio: list[float],
    cost_cfg: dict,
    prediction_error: float = 0.0,
) -> float:
    alpha = float(cost_cfg.get("alpha_estimation", 1.0))
    beta = float(cost_cfg.get("beta_prediction", 0.0))
    lam = float(cost_cfg.get("lambda_power", 0.1))
    eta = float(cost_cfg.get("eta_switch", 0.0))
    rho = float(cost_cfg.get("rho_coverage", 0.0))
    min_cov = float(cost_cfg.get("min_coverage_ratio", 0.0))
    unc = select_uncertainty_value(uncertainty_summary, cost_cfg)
    pred_loss = float(prediction_error)
    cov_pen = coverage_penalty(coverage_ratio, min_cov)
    return alpha * unc + beta * pred_loss + lam * power_cost + eta * switch_count + rho * cov_pen
