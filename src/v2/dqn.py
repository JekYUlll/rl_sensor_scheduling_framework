from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v2.custom_ppo import feasible_candidate_mask, oracle_greedy_candidate_index
from v2.env import WarmupEnvConfig, WarmupSchedulingEnv
from v2.oracle import LinearFrozenForecastOracle
from v2.power_projector import PowerConstraintsV2
from v2.rollout import RolloutResult, concat_rollout_results, rollout_metrics, run_policy_rollout
from v2.sensor_spec import SensorSpecV2


@dataclass(frozen=True)
class DQNConfig:
    total_timesteps: int = 100_000
    replay_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    n_step_return: int = 3
    hidden_dim: int = 128
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    exploration_fraction: float = 0.20
    max_grad_norm: float = 10.0
    event_start_prob: float = 0.67
    oracle_prefill_steps: int = 0
    oracle_prefill_lookahead_steps: int = 2
    device: str = "auto"
    seed: int = 42
    log_interval: int = 1_000
    history_path: str | None = None
    train_start_indices: tuple[int, ...] = ()
    train_start_min: int | None = None
    train_start_max: int | None = None


class ReplayBuffer:
    def __init__(self, *, size: int, obs_dim: int, n_actions: int, seed: int) -> None:
        self.size = int(size)
        self.obs = np.zeros((self.size, int(obs_dim)), dtype=np.float32)
        self.next_obs = np.zeros((self.size, int(obs_dim)), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)
        self.discounts = np.ones(self.size, dtype=np.float32)
        self.next_action_masks = np.ones((self.size, int(n_actions)), dtype=bool)
        self.pos = 0
        self.count = 0
        self.rng = np.random.default_rng(int(seed))

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
        discount: float,
    ) -> None:
        idx = int(self.pos)
        self.obs[idx] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.dones[idx] = float(done)
        self.discounts[idx] = float(discount)
        self.next_action_masks[idx] = np.asarray(next_action_mask, dtype=bool).reshape(-1)
        self.pos = (self.pos + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.count <= 0:
            raise RuntimeError("Cannot sample an empty replay buffer")
        idx = self.rng.integers(0, self.count, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "discounts": self.discounts[idx],
            "next_action_masks": self.next_action_masks[idx],
        }


class DQNTrainer:
    def __init__(
        self,
        *,
        truth_df: pd.DataFrame,
        sensor_specs: list[SensorSpecV2],
        constraints: PowerConstraintsV2,
        env_cfg: WarmupEnvConfig,
        oracle: LinearFrozenForecastOracle,
        candidate_masks: np.ndarray,
        cfg: DQNConfig,
    ) -> None:
        torch, nn = _torch_modules()
        self.truth_df = truth_df
        self.sensor_specs = list(sensor_specs)
        self.constraints = constraints
        self.env_cfg = env_cfg
        self.oracle = oracle
        self.candidate_masks_np = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(sensor_specs))
        if self.candidate_masks_np.shape[0] == 0:
            raise ValueError("candidate_masks must contain at least one action")
        self.cfg = cfg
        self.device = _select_device(torch, str(cfg.device))
        self.rng = np.random.default_rng(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))

        probe_env = self._make_env(seed_offset=0)
        probe_obs, _ = probe_env.reset()
        self.obs_dim = int(np.asarray(probe_obs).shape[0])
        self.q_net = _QNetwork(self.obs_dim, self.candidate_masks_np.shape[0], int(cfg.hidden_dim), nn).to(self.device)
        self.target_q_net = _QNetwork(self.obs_dim, self.candidate_masks_np.shape[0], int(cfg.hidden_dim), nn).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(cfg.learning_rate))
        self.replay = ReplayBuffer(
            size=int(cfg.replay_size),
            obs_dim=self.obs_dim,
            n_actions=self.candidate_masks_np.shape[0],
            seed=int(cfg.seed) + 1009,
        )
        self.history: list[dict[str, float | int]] = []

    def train(self) -> "DQNTrainer":
        total_timesteps = int(self.cfg.total_timesteps)
        if int(self.cfg.oracle_prefill_steps) > 0:
            self._prefill_replay_with_oracle(int(self.cfg.oracle_prefill_steps))
        env = self._make_env(seed_offset=0)
        obs, _ = env.reset(start_idx=self._sample_start_idx(int(self.env_cfg.episode_len or total_timesteps), seed_offset=0))
        recent_rewards: list[float] = []
        recent_losses: list[float] = []
        recent_actions: list[int] = []
        episode_rewards: list[float] = []
        episode_return = 0.0
        n_step_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque()

        for step in range(1, total_timesteps + 1):
            epsilon = self._epsilon(step)
            action_mask = feasible_candidate_mask(env, self.candidate_masks_np)
            action = self._select_action(obs, action_mask=action_mask, epsilon=epsilon)
            next_obs, reward, done, _ = env.step_mask(self.candidate_masks_np[action])
            next_action_mask = feasible_candidate_mask(env, self.candidate_masks_np)
            n_step_buffer.append((obs, int(action), float(reward), next_obs, bool(done), next_action_mask))
            if len(n_step_buffer) >= max(1, int(self.cfg.n_step_return)):
                self._add_n_step_transition(n_step_buffer)
                n_step_buffer.popleft()
            recent_rewards.append(float(reward))
            recent_actions.append(int(action))
            episode_return += float(reward)
            obs = next_obs

            if (
                step >= int(self.cfg.learning_starts)
                and self.replay.count > 0
                and step % max(1, int(self.cfg.train_freq)) == 0
            ):
                for _ in range(max(1, int(self.cfg.gradient_steps))):
                    recent_losses.append(self._update_once())
            if step % max(1, int(self.cfg.target_update_interval)) == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            if done:
                while n_step_buffer:
                    self._add_n_step_transition(n_step_buffer)
                    n_step_buffer.popleft()
                episode_rewards.append(float(episode_return))
                episode_return = 0.0
                env = self._make_env(seed_offset=step)
                obs, _ = env.reset(
                    start_idx=self._sample_start_idx(int(self.env_cfg.episode_len or total_timesteps), seed_offset=step)
                )

            if step == total_timesteps or step % max(1, int(self.cfg.log_interval)) == 0:
                row = {
                    "timesteps": int(step),
                    "epsilon": float(epsilon),
                    "reward_mean": _safe_mean(recent_rewards),
                    "loss": _safe_mean(recent_losses),
                    "episode_return_mean": _safe_mean(episode_rewards),
                    "unique_actions": int(np.unique(recent_actions).size) if recent_actions else 0,
                    "replay_size": int(self.replay.count),
                }
                self.history.append(row)
                self._flush_history()
                print(
                    "dqn_update "
                    f"timesteps={step} epsilon={float(epsilon):.4f} "
                    f"reward_mean={float(row['reward_mean']):.6f} "
                    f"loss={float(row['loss']):.6f} "
                    f"unique_actions={int(row['unique_actions'])}",
                    flush=True,
                )
                recent_rewards.clear()
                recent_losses.clear()
                recent_actions.clear()
                episode_rewards.clear()

        return self

    def _prefill_replay_with_oracle(self, total_steps: int) -> None:
        seed_base = 17_003
        env = self._make_env(seed_offset=seed_base)
        episode_len = int(self.env_cfg.episode_len or max(1, int(total_steps)))
        obs, _ = env.reset(start_idx=self._sample_start_idx(episode_len, seed_offset=seed_base))
        n_step_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque()
        recent_actions: list[int] = []
        for step in range(1, int(total_steps) + 1):
            action_mask = feasible_candidate_mask(env, self.candidate_masks_np)
            action = oracle_greedy_candidate_index(
                env,
                self.candidate_masks_np,
                lookahead_steps=max(1, int(self.cfg.oracle_prefill_lookahead_steps)),
            )
            if not bool(action_mask[int(action)]):
                valid = np.flatnonzero(action_mask)
                action = int(self.rng.choice(valid)) if valid.size else int(action)
            next_obs, reward, done, _ = env.step_mask(self.candidate_masks_np[int(action)])
            next_action_mask = feasible_candidate_mask(env, self.candidate_masks_np)
            n_step_buffer.append((obs, int(action), float(reward), next_obs, bool(done), next_action_mask))
            if len(n_step_buffer) >= max(1, int(self.cfg.n_step_return)):
                self._add_n_step_transition(n_step_buffer)
                n_step_buffer.popleft()
            recent_actions.append(int(action))
            obs = next_obs
            if done:
                while n_step_buffer:
                    self._add_n_step_transition(n_step_buffer)
                    n_step_buffer.popleft()
                env = self._make_env(seed_offset=seed_base + step)
                obs, _ = env.reset(start_idx=self._sample_start_idx(episode_len, seed_offset=seed_base + step))
        while n_step_buffer:
            self._add_n_step_transition(n_step_buffer)
            n_step_buffer.popleft()
        print(
            "dqn_oracle_prefill "
            f"steps={int(total_steps)} replay_size={int(self.replay.count)} "
            f"unique_actions={int(np.unique(recent_actions).size) if recent_actions else 0}",
            flush=True,
        )

    def predict_action(self, obs: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        torch, _ = _torch_modules()
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t).detach().cpu().numpy().reshape(-1)
        if action_mask is None:
            action_mask = np.ones(self.candidate_masks_np.shape[0], dtype=bool)
        masked_q = _masked_q_values(q_values, np.asarray(action_mask, dtype=bool))
        return int(np.argmax(masked_q))

    def predict_mask(self, obs: np.ndarray, action_mask: np.ndarray | None = None) -> np.ndarray:
        return self.candidate_masks_np[self.predict_action(obs, action_mask=action_mask)]

    def save(self, path: str | Path) -> None:
        torch, _ = _torch_modules()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.q_net.state_dict(),
                "target_state_dict": self.target_q_net.state_dict(),
                "cfg": asdict(self.cfg),
                "candidate_masks": self.candidate_masks_np,
                "obs_dim": self.obs_dim,
                "n_sensors": len(self.sensor_specs),
                "history": self.history,
            },
            str(out),
        )

    def save_history(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def _select_action(self, obs: np.ndarray, *, action_mask: np.ndarray, epsilon: float) -> int:
        valid = np.flatnonzero(np.asarray(action_mask, dtype=bool))
        if valid.size == 0:
            valid = np.arange(self.candidate_masks_np.shape[0], dtype=int)
        if self.rng.random() < float(epsilon):
            return int(self.rng.choice(valid))
        return self.predict_action(obs, action_mask=action_mask)

    def _update_once(self) -> float:
        torch, nn = _torch_modules()
        batch = self.replay.sample(max(1, int(self.cfg.batch_size)))
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(batch["discounts"], dtype=torch.float32, device=self.device)
        next_action_masks = torch.as_tensor(batch["next_action_masks"], dtype=torch.bool, device=self.device)

        q_values = self.q_net(obs).gather(1, actions.reshape(-1, 1)).reshape(-1)
        with torch.no_grad():
            next_online_q = self.q_net(next_obs).masked_fill(~next_action_masks, -1.0e9)
            next_actions = torch.argmax(next_online_q, dim=1)
            max_next_q = self.target_q_net(next_obs).gather(1, next_actions.reshape(-1, 1)).reshape(-1)
            target = rewards + discounts * (1.0 - dones) * max_next_q
        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), float(self.cfg.max_grad_norm))
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def _epsilon(self, step: int) -> float:
        decay_steps = max(1, int(float(self.cfg.exploration_fraction) * int(self.cfg.total_timesteps)))
        frac = min(1.0, max(0.0, float(step) / float(decay_steps)))
        return float(self.cfg.exploration_initial_eps) + frac * (
            float(self.cfg.exploration_final_eps) - float(self.cfg.exploration_initial_eps)
        )

    def _add_n_step_transition(
        self,
        buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]],
    ) -> None:
        if not buffer:
            return
        obs0, action0, _, _, _, _ = buffer[0]
        reward_sum = 0.0
        discount = 1.0
        next_obs = buffer[0][3]
        done = False
        next_action_mask = buffer[0][5]
        for _, _, reward, step_next_obs, step_done, step_next_action_mask in list(buffer)[: max(1, int(self.cfg.n_step_return))]:
            reward_sum += float(discount) * float(reward)
            next_obs = step_next_obs
            done = bool(step_done)
            next_action_mask = step_next_action_mask
            if done:
                discount = 0.0
                break
            discount *= float(self.cfg.gamma)
        self.replay.add(
            obs0,
            int(action0),
            float(reward_sum),
            next_obs,
            bool(done),
            next_action_mask,
            float(discount),
        )

    def _make_env(self, *, seed_offset: int) -> WarmupSchedulingEnv:
        return WarmupSchedulingEnv(
            self.truth_df,
            self.sensor_specs,
            self.constraints,
            replace(self.env_cfg, seed=int(self.env_cfg.seed) + int(seed_offset)),
            oracle=self.oracle,
        )

    def _sample_start_idx(self, steps: int, *, seed_offset: int) -> int:
        horizon = int(getattr(self.oracle.cfg, "horizon", 1))
        global_max_start = max(0, len(self.truth_df) - int(steps) - horizon - 1)
        min_start = max(0, int(self.cfg.train_start_min or 0))
        max_start = global_max_start
        if self.cfg.train_start_max is not None:
            max_start = min(max_start, int(self.cfg.train_start_max))
        if max_start < min_start:
            raise ValueError(f"No valid DQN training starts remain in [{min_start}, {max_start}]")
        rng = np.random.default_rng(int(self.cfg.seed) + int(seed_offset) + 71_239)
        if self.cfg.train_start_indices:
            starts = np.asarray(
                [int(x) for x in self.cfg.train_start_indices if min_start <= int(x) <= max_start],
                dtype=int,
            )
            if starts.size:
                return int(rng.choice(starts))
        event_flags = (
            self.truth_df[self.env_cfg.event_column].astype(bool).to_numpy()
            if self.env_cfg.event_column in self.truth_df.columns
            else np.zeros(len(self.truth_df), dtype=bool)
        )
        eligible_event_flags = np.zeros_like(event_flags, dtype=bool)
        eligible_event_flags[min_start : max_start + int(steps)] = event_flags[
            min_start : max_start + int(steps)
        ]
        event_indices = np.flatnonzero(eligible_event_flags)
        if event_indices.size and rng.random() < float(self.cfg.event_start_prob):
            event_idx = int(rng.choice(event_indices))
            return int(np.clip(event_idx - int(steps) // 3, min_start, max_start))
        return int(rng.integers(min_start, max_start + 1))

    def _flush_history(self) -> None:
        if self.cfg.history_path:
            self.save_history(self.cfg.history_path)


@dataclass
class DQNPolicy:
    trainer: DQNTrainer
    name: str = "dqn"

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        action_mask = feasible_candidate_mask(env, self.trainer.candidate_masks_np)
        return self.trainer.predict_mask(env._state().astype(np.float32), action_mask=action_mask)

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


def evaluate_dqn(
    *,
    trainer: DQNTrainer,
    truth_df: pd.DataFrame,
    sensor_specs: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle,
    steps: int,
    start_indices: tuple[int, ...],
) -> tuple[RolloutResult, dict[str, float | str | int]]:
    policy = DQNPolicy(trainer=trainer)
    rollouts = []
    for offset, start_idx in enumerate(start_indices):
        env = WarmupSchedulingEnv(
            truth_df,
            sensor_specs,
            constraints,
            replace(cfg, seed=int(cfg.seed) + int(offset)),
            oracle=oracle,
        )
        rollouts.append(run_policy_rollout(env, policy, steps=int(steps), start_idx=int(start_idx)))
    result = rollouts[0] if len(rollouts) == 1 else concat_rollout_results(rollouts, policy_name=policy.name)
    return result, rollout_metrics(result)


def _QNetwork(obs_dim: int, n_actions: int, hidden_dim: int, nn: Any) -> Any:
    class _DuelingQNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.LayerNorm(int(obs_dim)),
                nn.Linear(int(obs_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )
            self.value = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
            )
            self.advantage = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(n_actions)),
            )

        def forward(self, obs: Any) -> Any:
            features = self.features(obs)
            value = self.value(features)
            advantage = self.advantage(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)

    return _DuelingQNetwork()


def _masked_q_values(q_values: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    q = np.asarray(q_values, dtype=float).reshape(-1).copy()
    mask = np.asarray(action_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != q.shape[0] or not np.any(mask):
        return q
    q[~mask] = -1.0e12
    return q


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _torch_modules() -> tuple[Any, Any]:
    import torch
    from torch import nn

    return torch, nn


def _select_device(torch: Any, requested: str) -> Any:
    requested = str(requested)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
