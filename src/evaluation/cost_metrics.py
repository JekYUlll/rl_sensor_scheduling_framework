from __future__ import annotations

from scheduling.constraints import coverage_penalty


def compute_step_cost(
    uncertainty_trace: float,
    power_cost: float,
    switch_count: int,
    coverage_ratio: list[float],
    cost_cfg: dict,
) -> float:
    alpha = float(cost_cfg.get("alpha_estimation", 1.0))
    beta = float(cost_cfg.get("beta_prediction", 0.0))
    lam = float(cost_cfg.get("lambda_power", 0.1))
    eta = float(cost_cfg.get("eta_switch", 0.0))
    rho = float(cost_cfg.get("rho_coverage", 0.0))
    min_cov = float(cost_cfg.get("min_coverage_ratio", 0.0))
    pred_loss = 0.0
    cov_pen = coverage_penalty(coverage_ratio, min_cov)
    return alpha * uncertainty_trace + beta * pred_loss + lam * power_cost + eta * switch_count + rho * cov_pen
