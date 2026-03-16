from __future__ import annotations

import numpy as np
import torch
from torch import nn

from scheduling.rl.epsilon_scheduler import LinearEpsilonScheduler
from scheduling.rl.q_network import QNetwork
from scheduling.rl.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: dict, device: str = "cpu") -> None:
        hidden_dims = cfg.get("network", {}).get("hidden_dims", [128, 128])
        train_cfg = cfg.get("training", {})
        expl_cfg = cfg.get("exploration", {})

        self.device = torch.device(device)
        self.q = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_q = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
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

    def act(self, state_vec: np.ndarray, greedy: bool = False) -> int:
        epsilon = 0.0 if greedy else self.eps.value(self.total_steps)
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.q.net[-1].out_features))
        with torch.no_grad():
            s = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.q(s)
            return int(torch.argmax(qv, dim=1).item())

    def observe(self, state, action, reward, next_state, done) -> dict:
        self.replay.push(state, action, reward, next_state, done)
        self.total_steps += 1
        info = {"loss": None}
        if self.total_steps % self.train_interval == 0:
            loss = self._train_step()
            if loss is not None:
                info["loss"] = float(loss)
        if self.total_steps % self.target_update_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        return info

    def _train_step(self):
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None
        s, a, r, ns, d = self.replay.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.q(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_q(ns).max(dim=1, keepdim=True).values
            q_tgt = r + (1.0 - d) * self.gamma * q_next

        loss = self.loss_fn(q_pred, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        return loss.item()

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q.load_state_dict(self.q.state_dict())
