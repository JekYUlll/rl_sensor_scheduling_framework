from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from scheduling.online_projector import OnlineSubsetProjector
from scheduling.rl.epsilon_scheduler import LinearEpsilonScheduler
from scheduling.rl.q_network import BranchingQNetwork
from scheduling.rl.replay_buffer import ReplayBuffer


class ScoreDQNAgent:
    """Factorized DQN over sensor on/off decisions."""

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

        self.q = BranchingQNetwork(state_dim, len(self.sensor_ids), hidden_dims).to(self.device)
        self.target_q = BranchingQNetwork(state_dim, len(self.sensor_ids), hidden_dims).to(self.device)
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

    def _q_values(self, state_vec: np.ndarray, network: BranchingQNetwork) -> torch.Tensor:
        s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        return network(s).squeeze(0)

    def _subset_value_from_q(self, q_branch: torch.Tensor, subset: list[str]) -> torch.Tensor:
        mask = torch.zeros(len(self.sensor_ids), dtype=torch.bool, device=q_branch.device)
        for sid in subset:
            idx = self.sensor_to_idx.get(str(sid))
            if idx is not None:
                mask[idx] = True
        return q_branch[mask, 1].sum() + q_branch[~mask, 0].sum()

    def _best_subset(self, q_branch: torch.Tensor, prev_selected: list[str] | None = None) -> tuple[list[str], float]:
        masks_np, feasible = self._candidate_masks(prev_selected)
        masks = torch.as_tensor(masks_np, dtype=torch.float32, device=q_branch.device)
        delta = q_branch[:, 1] - q_branch[:, 0]
        off_sum = q_branch[:, 0].sum()
        values = off_sum + masks @ delta
        sizes = masks.sum(dim=1)
        best_idx = int(torch.argmax(values + 1e-6 * sizes).item())
        subset = list(feasible[best_idx])
        value = float(values[best_idx].item())
        return subset, value

    def act(self, state_vec: np.ndarray, greedy: bool = False, prev_selected: list[str] | None = None) -> list[str]:
        epsilon = 0.0 if greedy else self.eps.value(self.total_steps)
        if np.random.rand() < epsilon:
            return self.projector.sample_random_subset(prev_selected=prev_selected)
        with torch.no_grad():
            q_branch = self._q_values(state_vec, self.q)
            subset, _ = self._best_subset(q_branch, prev_selected=prev_selected)
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

        q_pred_all = self.q(s_t)
        q_pred = torch.where(a_mask_t, q_pred_all[:, :, 1], q_pred_all[:, :, 0]).sum(dim=1)

        with torch.no_grad():
            q_next_all = self.target_q(ns_t)
            q_next_vals = torch.zeros(q_next_all.shape[0], dtype=torch.float32, device=self.device)
            for sample_idx in range(q_next_all.shape[0]):
                prev_selected = [sid for idx, sid in enumerate(self.sensor_ids) if bool(a_mask_t[sample_idx, idx].item())]
                _, q_val = self._best_subset(q_next_all[sample_idx], prev_selected=prev_selected)
                q_next_vals[sample_idx] = float(q_val)
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
    ) -> None:
        super().__init__(state_dim=state_dim, sensor_ids=sensor_ids, cfg=cfg, projector=projector, device=device)
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
            self.power_reference = max(float(cmdp_cfg.get("power_reference", projector.per_step_budget)), 1e-6)
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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)
