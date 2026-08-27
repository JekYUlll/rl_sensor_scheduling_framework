"""Mask-structured forecast-cost regression for flexible sensor subsets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MaskCostRegressorConfig:
    context_dim: int
    sensor_count: int
    context_hidden_dim: int = 96
    sensor_embedding_dim: int = 32
    action_hidden_dim: int = 96


class MaskCostRegressor(nn.Module):
    """Predict candidate costs with parameters shared across sensor subsets."""

    def __init__(self, cfg: MaskCostRegressorConfig) -> None:
        super().__init__()
        if cfg.context_dim <= 0 or cfg.sensor_count <= 0:
            raise ValueError("context_dim and sensor_count must be positive")
        self.cfg = cfg
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(cfg.context_dim),
            nn.Linear(cfg.context_dim, cfg.context_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.context_hidden_dim, cfg.context_hidden_dim),
            nn.GELU(),
        )
        self.sensor_embeddings = nn.Parameter(
            torch.empty(cfg.sensor_count, cfg.sensor_embedding_dim)
        )
        nn.init.normal_(self.sensor_embeddings, mean=0.0, std=0.08)
        self.context_to_sensor = nn.Linear(
            cfg.context_hidden_dim,
            cfg.sensor_count * cfg.sensor_embedding_dim,
        )
        action_feature_dim = (
            cfg.context_hidden_dim
            + 2 * cfg.sensor_embedding_dim
            + cfg.sensor_count
            + 2
        )
        self.cost_head = nn.Sequential(
            nn.LayerNorm(action_feature_dim),
            nn.Linear(action_feature_dim, cfg.action_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.action_hidden_dim, cfg.action_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.action_hidden_dim, 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_cost_features: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError("context must be rank-2")
        if candidate_masks.ndim != 2:
            raise ValueError("candidate_masks must be rank-2")
        if candidate_masks.shape[1] != self.cfg.sensor_count:
            raise ValueError("candidate mask width must match sensor_count")
        if candidate_cost_features.shape != (candidate_masks.shape[0], 2):
            raise ValueError("candidate_cost_features must have shape [actions, 2]")

        masks = candidate_masks.to(dtype=context.dtype)
        batch_size = context.shape[0]
        action_count = masks.shape[0]
        encoded = self.context_encoder(context)
        conditional_sensor = self.context_to_sensor(encoded).reshape(
            batch_size,
            self.cfg.sensor_count,
            self.cfg.sensor_embedding_dim,
        )
        base_sensor = self.sensor_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        selected_count = masks.sum(dim=1).clamp_min(1.0)
        pooled_base = torch.einsum("as,bsd->bad", masks, base_sensor)
        pooled_conditional = torch.einsum("as,bsd->bad", masks, conditional_sensor)
        divisor = selected_count.reshape(1, action_count, 1)
        pooled_base = pooled_base / divisor
        pooled_conditional = pooled_conditional / divisor
        expanded_context = encoded.unsqueeze(1).expand(-1, action_count, -1)
        expanded_masks = masks.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_costs = candidate_cost_features.unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat(
            [
                expanded_context,
                pooled_base,
                pooled_conditional,
                expanded_masks,
                expanded_costs,
            ],
            dim=-1,
        )
        return self.cost_head(features).squeeze(-1)
