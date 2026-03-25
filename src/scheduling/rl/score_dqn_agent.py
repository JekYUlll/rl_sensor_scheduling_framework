from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from scheduling.online_projector import OnlineSubsetProjector
from scheduling.rl.epsilon_scheduler import LinearEpsilonScheduler
from scheduling.rl.q_network import SubsetQNetwork
from scheduling.rl.replay_buffer import ConstraintReplayBuffer, ReplayBuffer


class ScoreDQNAgent:
    """DQN over online-generated feasible sensor subsets."""

    def __init__(self, state_dim: int, sensor_ids: list[str], cfg: dict, projector: OnlineSubsetProjector, device: str | None = None) -> None:
        hidden_dims = cfg.get("network", {}).get("hidden_dims", [128, 128])
        train_cfg = cfg.get("training", {})
        expl_cfg = cfg.get("exploration", {})

        resolved_device = device or str(cfg.get("device", "cpu"))
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved_device)
        self.sensor_ids = [str(sid) for sid in sensor_ids]
        self.sensor_to_idx = {sid: i for i, sid in enumerate(self.sensor_ids)}
        self.projector = projector
        self._subset_mask_cache: dict[tuple[str, ...], tuple[np.ndarray, list[tuple[str, ...]]]] = {}

        self.q = SubsetQNetwork(state_dim, len(self.sensor_ids), hidden_dims).to(self.device)
        self.target_q = SubsetQNetwork(state_dim, len(self.sensor_ids), hidden_dims).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.gamma = float(train_cfg.get("gamma", 0.99))
        self.batch_size = int(train_cfg.get("batch_size", 64))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 500))
        self.target_update_interval = int(train_cfg.get("target_update_interval", 200))
        self.train_interval = int(train_cfg.get("train_interval", 1))
        self.grad_clip = float(train_cfg.get("grad_clip", 5.0))

        lr = float(train_cfg.get("lr", 1e-3))
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        replay_size = int(train_cfg.get("replay_size", 50000))
        self.replay = ReplayBuffer(replay_size)
        self.eps = LinearEpsilonScheduler(
            eps_start=float(expl_cfg.get("eps_start", 1.0)),
            eps_end=float(expl_cfg.get("eps_end", 0.05)),
            decay_steps=int(expl_cfg.get("eps_decay_steps", 10000)),
        )
        self.total_steps = 0

    def _mask_from_subset(self, subset: list[str]) -> np.ndarray:
        mask = np.zeros(len(self.sensor_ids), dtype=np.int64)
        for sid in subset:
            idx = self.sensor_to_idx.get(str(sid))
            if idx is not None:
                mask[idx] = 1
        return mask

    def _candidate_masks(self, prev_selected: list[str] | None = None) -> tuple[np.ndarray, list[tuple[str, ...]]]:
        key = tuple(sorted(str(sid) for sid in (prev_selected or [])))
        cached = self._subset_mask_cache.get(key)
        if cached is not None:
            return cached
        feasible = [tuple(str(sid) for sid in subset) for subset in self.projector.feasible_subsets(prev_selected)]
        if not feasible:
            feasible = [tuple()]
        masks = np.zeros((len(feasible), len(self.sensor_ids)), dtype=np.float32)
        for row, subset in enumerate(feasible):
            for sid in subset:
                masks[row, self.sensor_to_idx[sid]] = 1.0
        cached = (masks, feasible)
        self._subset_mask_cache[key] = cached
        return cached

    def _score_masks(self, state_vec: np.ndarray, masks_np: np.ndarray, network: SubsetQNetwork) -> torch.Tensor:
        s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        masks = torch.as_tensor(masks_np, dtype=torch.float32, device=self.device)
        return network(s, masks)

    def _best_subset(
        self,
        state_vec: np.ndarray,
        network: SubsetQNetwork,
        prev_selected: list[str] | None = None,
    ) -> tuple[list[str], np.ndarray, float]:
        masks_np, feasible = self._candidate_masks(prev_selected)
        values = self._score_masks(state_vec, masks_np, network)
        sizes = torch.as_tensor(masks_np.sum(axis=1), dtype=torch.float32, device=values.device)
        best_idx = int(torch.argmax(values + 1e-6 * sizes).item())
        subset = list(feasible[best_idx])
        value = float(values[best_idx].item())
        return subset, masks_np[best_idx], value

    def act(self, state_vec: np.ndarray, greedy: bool = False, prev_selected: list[str] | None = None) -> list[str]:
        epsilon = 0.0 if greedy else self.eps.value(self.total_steps)
        if np.random.rand() < epsilon:
            return self.projector.sample_random_subset(prev_selected=prev_selected)
        with torch.no_grad():
            subset, _, _ = self._best_subset(state_vec, self.q, prev_selected=prev_selected)
            return subset

    def observe(
        self,
        state: np.ndarray,
        action_subset: list[str],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict[str, float | None]:
        self.replay.push(state, self._mask_from_subset(action_subset), reward, next_state, done)
        self.total_steps += 1
        info: dict[str, float | None] = {"loss": None}
        if self.total_steps % self.train_interval == 0:
            loss = self._train_step()
            if loss is not None:
                info["loss"] = loss
        if self.total_steps % self.target_update_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        return info

    def _train_step(self) -> float | None:
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None
        s, a_mask, r, ns, d = self.replay.sample(self.batch_size)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_mask_t = torch.as_tensor(a_mask, dtype=torch.bool, device=self.device)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        ns_t = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        q_pred = self.q(s_t, a_mask_t.float())

        with torch.no_grad():
            q_next_vals = torch.zeros(ns_t.shape[0], dtype=torch.float32, device=self.device)
            for sample_idx in range(ns_t.shape[0]):
                prev_selected = [sid for idx, sid in enumerate(self.sensor_ids) if bool(a_mask_t[sample_idx, idx].item())]
                subset, best_mask_np, _ = self._best_subset(
                    ns[sample_idx],
                    self.q,
                    prev_selected=prev_selected,
                )
                _ = subset
                best_mask_t = torch.as_tensor(best_mask_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_next_vals[sample_idx] = self.target_q(ns_t[sample_idx].unsqueeze(0), best_mask_t).squeeze(0)
            q_next = q_next_vals
            q_tgt = r_t + (1.0 - d_t) * self.gamma * q_next

        loss = self.loss_fn(q_pred, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q.load_state_dict(self.q.state_dict())


class ConstrainedScoreDQNAgent(ScoreDQNAgent):
    def __init__(
        self,
        state_dim: int,
        sensor_ids: list[str],
        cfg: dict,
        projector: OnlineSubsetProjector,
        device: str | None = None,
        episode_len: int | None = None,
    ) -> None:
        super().__init__(state_dim=state_dim, sensor_ids=sensor_ids, cfg=cfg, projector=projector, device=device)
        cmdp_cfg = cfg.get("cmdp", {})
        train_cfg = cfg.get("training", {})
        hidden_dims = cfg.get("network", {}).get("hidden_dims", [128, 128])

        self.cost_q = SubsetQNetwork(state_dim, len(self.sensor_ids), hidden_dims).to(self.device)
        self.target_cost_q = SubsetQNetwork(state_dim, len(self.sensor_ids), hidden_dims).to(self.device)
        self.target_cost_q.load_state_dict(self.cost_q.state_dict())
        self.cost_optimizer = torch.optim.Adam(self.cost_q.parameters(), lr=float(train_cfg.get("lr", 1e-3)))

        replay_size = int(train_cfg.get("replay_size", 50000))
        self.replay = ConstraintReplayBuffer(replay_size)
        self.average_power_budget = _optional_float(cmdp_cfg.get("average_power_budget"))
        self.episode_energy_budget = _optional_float(cmdp_cfg.get("episode_energy_budget"))
        self.lambda_cost = float(cmdp_cfg.get("lambda_cost_init", cmdp_cfg.get("lambda_avg_init", 0.0)))
        self.dual_lr = float(cmdp_cfg.get("dual_lr_cost", cmdp_cfg.get("dual_lr_avg", 0.05)))
        self.lambda_max = float(cmdp_cfg.get("lambda_max", 100.0))
        self.violation_ema_beta = float(cmdp_cfg.get("violation_ema_beta", 0.9))
        self.violation_ema = 0.0
        self.normalize_power = bool(cmdp_cfg.get("normalize_power", True))
        if self.normalize_power:
            self.power_reference = max(float(cmdp_cfg.get("power_reference", projector.per_step_budget)), 1e-6)
        else:
            self.power_reference = 1.0
        self.episode_len = episode_len or (int(cmdp_cfg.get("episode_len", 0)) or None)
        self.effective_average_power_budget = _effective_average_power_budget(
            average_power_budget=self.average_power_budget,
            episode_energy_budget=self.episode_energy_budget,
            episode_len=self.episode_len,
        )

    def _norm_power(self, power: float) -> float:
        return float(power) / self.power_reference

    def _effective_budget(self) -> float | None:
        if self.effective_average_power_budget is None:
            return None
        if self.normalize_power:
            return float(self.effective_average_power_budget) / self.power_reference
        return float(self.effective_average_power_budget)

    def shape_reward(self, task_reward: float, steady_power: float) -> float:
        return float(task_reward - self.lambda_cost * self._norm_power(steady_power))

    def act(self, state_vec: np.ndarray, greedy: bool = False, prev_selected: list[str] | None = None) -> list[str]:
        epsilon = 0.0 if greedy else self.eps.value(self.total_steps)
        if np.random.rand() < epsilon:
            return self.projector.sample_random_subset(prev_selected=prev_selected)
        with torch.no_grad():
            masks_np, feasible = self._candidate_masks(prev_selected)
            reward_vals = self._score_masks(state_vec, masks_np, self.q)
            cost_vals = self._score_masks(state_vec, masks_np, self.cost_q)
            lag_vals = reward_vals - self.lambda_cost * cost_vals
            sizes = torch.as_tensor(masks_np.sum(axis=1), dtype=torch.float32, device=lag_vals.device)
            best_idx = int(torch.argmax(lag_vals + 1e-6 * sizes).item())
            subset = list(feasible[best_idx])
            return subset

    def observe(
        self,
        state: np.ndarray,
        action_subset: list[str],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        constraint_cost: float | None = None,
    ) -> dict[str, float | None]:
        self.replay.push(
            state,
            self._mask_from_subset(action_subset),
            reward,
            self._norm_power(float(constraint_cost or 0.0)),
            next_state,
            done,
        )
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
        s, a_mask, task_r, cost_r, ns, d = self.replay.sample(self.batch_size)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_mask_t = torch.as_tensor(a_mask, dtype=torch.bool, device=self.device)
        task_r_t = torch.as_tensor(task_r, dtype=torch.float32, device=self.device)
        cost_r_t = torch.as_tensor(cost_r, dtype=torch.float32, device=self.device)
        ns_t = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        reward_pred = self.q(s_t, a_mask_t.float())
        cost_pred = self.cost_q(s_t, a_mask_t.float())

        with torch.no_grad():
            reward_next_vals = torch.zeros(ns_t.shape[0], dtype=torch.float32, device=self.device)
            cost_next_vals = torch.zeros_like(reward_next_vals)
            for sample_idx in range(ns_t.shape[0]):
                prev_selected = [sid for idx, sid in enumerate(self.sensor_ids) if bool(a_mask_t[sample_idx, idx].item())]
                masks_np, feasible = self._candidate_masks(prev_selected)
                reward_vals = self._score_masks(ns[sample_idx], masks_np, self.q)
                cost_vals = self._score_masks(ns[sample_idx], masks_np, self.cost_q)
                lag_vals = reward_vals - self.lambda_cost * cost_vals
                sizes = torch.as_tensor(masks_np.sum(axis=1), dtype=torch.float32, device=lag_vals.device)
                best_idx = int(torch.argmax(lag_vals + 1e-6 * sizes).item())
                _ = feasible[best_idx]
                best_mask_t = torch.as_tensor(masks_np[best_idx], dtype=torch.float32, device=self.device).unsqueeze(0)
                reward_next_vals[sample_idx] = self.target_q(ns_t[sample_idx].unsqueeze(0), best_mask_t).squeeze(0)
                cost_next_vals[sample_idx] = self.target_cost_q(ns_t[sample_idx].unsqueeze(0), best_mask_t).squeeze(0)
            reward_tgt = task_r_t + (1.0 - d_t) * self.gamma * reward_next_vals
            cost_tgt = cost_r_t + (1.0 - d_t) * self.gamma * cost_next_vals

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

    def end_episode(self, mean_power: float, total_energy: float) -> dict[str, float]:
        raw_avg_violation = 0.0
        if self.effective_average_power_budget is not None:
            raw_avg_violation = float(mean_power - self.effective_average_power_budget)
            effective_budget = self._effective_budget()
            assert effective_budget is not None
            violation = self._norm_power(mean_power) - effective_budget
            self.violation_ema = self.violation_ema_beta * self.violation_ema + (1.0 - self.violation_ema_beta) * violation
            self.lambda_cost = min(self.lambda_max, max(0.0, self.lambda_cost + self.dual_lr * self.violation_ema))
        return {
            "lambda_avg": float(self.lambda_cost),
            "lambda_energy": 0.0,
            "lambda_cost": float(self.lambda_cost),
            "avg_power_violation": raw_avg_violation,
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
        super().load(path)


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
