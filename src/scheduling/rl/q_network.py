from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [state_dim] + list(hidden_dims) + [action_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchingQNetwork(nn.Module):
    def __init__(self, state_dim: int, num_sensors: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [state_dim] + list(hidden_dims)
        trunk: list[nn.Module] = []
        for i in range(len(dims) - 1):
            trunk.append(nn.Linear(dims[i], dims[i + 1]))
            trunk.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk)
        last_dim = dims[-1] if hidden_dims else state_dim
        self.on_head = nn.Linear(last_dim, num_sensors)
        self.off_head = nn.Linear(last_dim, num_sensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        off = self.off_head(z)
        on = self.on_head(z)
        return torch.stack([off, on], dim=-1)


class SubsetQNetwork(nn.Module):
    """Score a concrete sensor subset mask conditioned on the RL state."""

    def __init__(self, state_dim: int, num_sensors: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [state_dim + num_sensors] + list(hidden_dims) + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
        if state.shape[0] == 1 and action_mask.shape[0] > 1:
            state = state.expand(action_mask.shape[0], -1)
        x = torch.cat([state, action_mask], dim=-1)
        return self.net(x).squeeze(-1)
