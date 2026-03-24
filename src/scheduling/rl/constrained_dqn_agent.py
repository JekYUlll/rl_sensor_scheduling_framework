from __future__ import annotations

from scheduling.rl.dqn_agent import DQNAgent


class ConstrainedDQNAgent(DQNAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: dict,
        device: str | None = None,
        steady_power_limit: float | None = None,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, cfg=cfg, device=device)
        cmdp_cfg = cfg.get("cmdp", {})
        self.average_power_budget = _optional_float(cmdp_cfg.get("average_power_budget"))
        self.episode_energy_budget = _optional_float(cmdp_cfg.get("episode_energy_budget"))
        self.lambda_avg = float(cmdp_cfg.get("lambda_avg_init", 0.0))
        self.lambda_energy = float(cmdp_cfg.get("lambda_energy_init", 0.0))
        self.dual_lr_avg = float(cmdp_cfg.get("dual_lr_avg", 0.05))
        self.dual_lr_energy = float(cmdp_cfg.get("dual_lr_energy", 0.001))
        self.lambda_max = float(cmdp_cfg.get("lambda_max", 100.0))
        self.normalize_power = bool(cmdp_cfg.get("normalize_power", True))
        if self.normalize_power:
            self.power_reference = max(float(cmdp_cfg.get("power_reference", steady_power_limit or 1.0)), 1e-6)
        else:
            self.power_reference = 1.0

    def _norm_power(self, power: float) -> float:
        return float(power) / self.power_reference

    def shape_reward(self, task_reward: float, steady_power: float) -> float:
        power_term = self._norm_power(steady_power)
        return float(task_reward - self.lambda_avg * power_term - self.lambda_energy * power_term)

    def end_episode(self, mean_power: float, total_energy: float) -> dict[str, float]:
        metrics: dict[str, float] = {
            "lambda_avg": float(self.lambda_avg),
            "lambda_energy": float(self.lambda_energy),
            "avg_power_violation": 0.0,
            "energy_violation": 0.0,
        }
        if self.average_power_budget is not None:
            violation = float(mean_power - self.average_power_budget)
            self.lambda_avg = min(self.lambda_max, max(0.0, self.lambda_avg + self.dual_lr_avg * violation))
            metrics["avg_power_violation"] = violation
            metrics["lambda_avg"] = float(self.lambda_avg)
        if self.episode_energy_budget is not None:
            violation = float(total_energy - self.episode_energy_budget)
            self.lambda_energy = min(self.lambda_max, max(0.0, self.lambda_energy + self.dual_lr_energy * violation))
            metrics["energy_violation"] = violation
            metrics["lambda_energy"] = float(self.lambda_energy)
        return metrics


def _optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)
