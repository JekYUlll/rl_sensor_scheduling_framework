from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from scheduling.rl.dqn_agent import DQNAgent
from scheduling.rl.q_network import QNetwork
from scheduling.rl.replay_buffer import ConstraintReplayBuffer


class ConstrainedDQNAgent(DQNAgent):
    """Primal-dual DQN with separate reward and cost critics."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: dict,
        device: str | None = None,
        steady_power_limit: float | None = None,
        episode_len: int | None = None,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, cfg=cfg, device=device)
        hidden_dims = cfg.get("network", {}).get("hidden_dims", [128, 128])
        train_cfg = cfg.get("training", {})
        cmdp_cfg = cfg.get("cmdp", {})

        self.cost_q = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_cost_q = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_cost_q.load_state_dict(self.cost_q.state_dict())
        self.cost_optimizer = torch.optim.Adam(self.cost_q.parameters(), lr=float(train_cfg.get("lr", 1e-3)))

        replay_size = int(train_cfg.get("replay_size", 50000))
        self.replay = ConstraintReplayBuffer(replay_size)
        self.lambda_cost = float(cmdp_cfg.get("lambda_cost_init", cmdp_cfg.get("lambda_avg_init", 0.0)))
        self.dual_lr = float(cmdp_cfg.get("dual_lr_cost", cmdp_cfg.get("dual_lr_avg", 0.05)))
        self.lambda_max = float(cmdp_cfg.get("lambda_max", 100.0))
        self.violation_ema_beta = float(cmdp_cfg.get("violation_ema_beta", 0.9))
        self.violation_ema = 0.0
        self.normalize_power = bool(cmdp_cfg.get("normalize_power", True))
        self.power_reference = max(float(cmdp_cfg.get("power_reference", steady_power_limit or 1.0)), 1e-6)

        average_power_budget = _optional_float(cmdp_cfg.get("average_power_budget"))
        episode_energy_budget = _optional_float(cmdp_cfg.get("episode_energy_budget"))
        self.effective_average_power_budget = _effective_average_power_budget(
            average_power_budget=average_power_budget,
            episode_energy_budget=episode_energy_budget,
            episode_len=episode_len,
        )
        self.average_power_budget = average_power_budget
        self.episode_energy_budget = episode_energy_budget
        self.episode_len = episode_len

    def _norm_power(self, power: float) -> float:
        return float(power) / self.power_reference if self.normalize_power else float(power)

    def _effective_budget(self) -> float | None:
        if self.effective_average_power_budget is None:
            return None
        if self.normalize_power:
            return float(self.effective_average_power_budget) / self.power_reference
        return float(self.effective_average_power_budget)

    def _lagrangian_q(self, reward_q: torch.Tensor, cost_q: torch.Tensor) -> torch.Tensor:
        return reward_q - self.lambda_cost * cost_q

    def act(self, state_vec: np.ndarray, greedy: bool = False, feasible_action_ids: list[int] | None = None) -> int:
        feasible = self._resolve_feasible(feasible_action_ids)
        if not feasible:
            return 0
        epsilon = 0.0 if greedy else self.eps.value(self.total_steps)
        if np.random.rand() < epsilon:
            return int(np.random.choice(feasible))
        with torch.no_grad():
            s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            reward_q = self.q(s).squeeze(0)
            cost_q = self.cost_q(s).squeeze(0)
            lag_q = self._lagrangian_q(reward_q, cost_q)
            feasible_tensor = torch.as_tensor(feasible, dtype=torch.int64, device=self.device)
            feasible_q = lag_q.index_select(0, feasible_tensor)
            best_idx = int(torch.argmax(feasible_q).item())
            return int(feasible[best_idx])

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        constraint_cost: float | None = None,
    ) -> dict[str, float | None]:
        self.replay.push(state, action, reward, self._norm_power(float(constraint_cost or 0.0)), next_state, done)
        self.total_steps += 1
        info: dict[str, float | None] = {"loss": None}
        if self.total_steps % self.train_interval == 0:
            loss = self._train_step()
            if loss is not None:
                info["loss"] = loss
        if self.total_steps % self.target_update_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())
            self.target_cost_q.load_state_dict(self.cost_q.state_dict())
        return info

    def _train_step(self) -> float | None:
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None
        s, a, task_r, cost_r, ns, d = self.replay.sample(self.batch_size)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        task_r_t = torch.as_tensor(task_r, dtype=torch.float32, device=self.device).unsqueeze(1)
        cost_r_t = torch.as_tensor(cost_r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        reward_pred = self.q(s_t).gather(1, a_t)
        cost_pred = self.cost_q(s_t).gather(1, a_t)

        with torch.no_grad():
            reward_next_online = self.q(ns_t)
            cost_next_online = self.cost_q(ns_t)
            lag_next_online = self._lagrangian_q(reward_next_online, cost_next_online)
            next_actions = lag_next_online.argmax(dim=1, keepdim=True)
            reward_next = self.target_q(ns_t).gather(1, next_actions)
            cost_next = self.target_cost_q(ns_t).gather(1, next_actions)
            reward_tgt = task_r_t + (1.0 - d_t) * self.gamma * reward_next
            cost_tgt = cost_r_t + (1.0 - d_t) * self.gamma * cost_next

        reward_loss = self.loss_fn(reward_pred, reward_tgt)
        cost_loss = self.loss_fn(cost_pred, cost_tgt)

        self.optimizer.zero_grad()
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_q.parameters(), max_norm=self.grad_clip)
        self.cost_optimizer.step()
        return float((reward_loss + cost_loss).item())

    def shape_reward(self, task_reward: float, steady_power: float) -> float:
        return float(task_reward - self.lambda_cost * self._norm_power(steady_power))

    def end_episode(self, mean_power: float, total_energy: float) -> dict[str, float]:
        violation = 0.0
        effective_budget = self.effective_average_power_budget
        if effective_budget is not None:
            effective_mean_power = self._norm_power(mean_power)
            effective_budget = self._effective_budget()
            assert effective_budget is not None
            violation = float(effective_mean_power - effective_budget)
            self.violation_ema = self.violation_ema_beta * self.violation_ema + (1.0 - self.violation_ema_beta) * violation
            self.lambda_cost = min(self.lambda_max, max(0.0, self.lambda_cost + self.dual_lr * self.violation_ema))
        return {
            "lambda_avg": float(self.lambda_cost),
            "lambda_energy": 0.0,
            "lambda_cost": float(self.lambda_cost),
            "avg_power_violation": float(mean_power - (self.effective_average_power_budget or mean_power)),
            "energy_violation": float(total_energy - (self.episode_energy_budget or total_energy)),
            "effective_average_power_budget": float(self.effective_average_power_budget or 0.0),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "reward_q": self.q.state_dict(),
                "cost_q": self.cost_q.state_dict(),
                "lambda_cost": float(self.lambda_cost),
                "violation_ema": float(self.violation_ema),
            },
            path,
        )

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict) and "reward_q" in payload:
            self.q.load_state_dict(payload["reward_q"])
            self.target_q.load_state_dict(self.q.state_dict())
            self.cost_q.load_state_dict(payload["cost_q"])
            self.target_cost_q.load_state_dict(self.cost_q.state_dict())
            self.lambda_cost = float(payload.get("lambda_cost", self.lambda_cost))
            self.violation_ema = float(payload.get("violation_ema", self.violation_ema))
            return
        self.q.load_state_dict(payload)
        self.target_q.load_state_dict(self.q.state_dict())


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _effective_average_power_budget(
    average_power_budget: float | None,
    episode_energy_budget: float | None,
    episode_len: int | None,
) -> float | None:
    candidates: list[float] = []
    if average_power_budget is not None:
        candidates.append(float(average_power_budget))
    if episode_energy_budget is not None and episode_len and episode_len > 0:
        candidates.append(float(episode_energy_budget) / float(episode_len))
    if not candidates:
        return None
    return min(candidates)
