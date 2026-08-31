from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv
from v2.mask_cost_regressor import MaskCostRegressor, MaskCostRegressorConfig
from v2.oracle import LinearFrozenForecastOracle
from v2.policies import MinDwellPolicyWrapper
from v2.power_projector import PowerConstraintsV2
from v2.rollout import RolloutResult, concat_rollout_results, rollout_metrics, run_policy_rollout
from v2.sensor_spec import SensorSpecV2


def full_rollout_schedule(total_timesteps: int, n_steps: int) -> tuple[int, ...]:
    """Return full PPO rollout sizes that meet or exceed the sample budget."""
    total = max(0, int(total_timesteps))
    rollout = int(n_steps)
    if rollout <= 0:
        raise ValueError("n_steps must be positive")
    count = (total + rollout - 1) // rollout
    return tuple(rollout for _ in range(count))


def policy_decision_mask(action_masks: np.ndarray) -> np.ndarray:
    """Mark rollout rows where the policy had more than one executable action."""
    masks = np.asarray(action_masks, dtype=bool)
    if masks.ndim != 2:
        raise ValueError("action_masks must be a rank-2 array")
    return (np.sum(masks, axis=1) > 1).astype(np.float32)


def compute_decision_block_credit(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    decision_rows: np.ndarray,
    episode_ids: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    reward_mode: str = "sum",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute semi-Markov advantages for rows where a new action was possible.

    Each decision receives a reward defined over the interval until the next
    decision in the same episode. ``sum`` uses the discounted block sum;
    ``terminal`` uses only the block-end loss, avoiding repeated credit for
    overlapping forecast windows. Forced dwell rows remain available to the
    critic, but do not become separate policy transitions.
    """
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    decisions = np.asarray(decision_rows, dtype=np.float32).reshape(-1) > 0.5
    episodes = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    n = int(rewards.size)
    if not (values.size == dones.size == decisions.size == episodes.size == n):
        raise ValueError("reward, value, done, decision, and episode arrays must have equal length")
    mode = str(reward_mode).strip().lower()
    if mode not in {"sum", "terminal"}:
        raise ValueError("reward_mode must be 'sum' or 'terminal'")
    block_advantages = np.zeros(n, dtype=np.float32)
    block_returns = np.zeros(n, dtype=np.float32)
    durations = np.zeros(n, dtype=np.int64)
    decision_indices = np.flatnonzero(decisions)
    if decision_indices.size == 0:
        return block_advantages, block_returns, durations

    deltas: list[float] = []
    block_indices: list[int] = []
    block_discounts: list[float] = []
    for position, start in enumerate(decision_indices.tolist()):
        episode = int(episodes[start])
        next_start = n
        if position + 1 < decision_indices.size:
            candidate = int(decision_indices[position + 1])
            if int(episodes[candidate]) == episode:
                next_start = candidate
        block_reward = 0.0
        discount = 1.0
        terminal_row = int(start)
        for row in range(int(start), int(next_start)):
            block_reward += discount * float(rewards[row])
            discount *= float(gamma)
            terminal_row = int(row)
            if float(dones[row]) > 0.5:
                next_start = row + 1
                break
        duration = max(1, int(next_start) - int(start))
        if mode == "terminal":
            block_reward = (float(gamma) ** max(0, duration - 1)) * float(rewards[terminal_row])
        next_value = (
            float(values[next_start])
            if next_start < n and int(episodes[next_start]) == episode
            else 0.0
        )
        transition_discount = float(gamma) ** duration
        block_indices.append(int(start))
        block_discounts.append(transition_discount)
        durations[start] = int(duration)
        deltas.append(block_reward + transition_discount * next_value - float(values[start]))

    next_advantage = 0.0
    previous_episode: int | None = None
    for reverse_position in range(len(block_indices) - 1, -1, -1):
        start = block_indices[reverse_position]
        episode = int(episodes[start])
        if previous_episode is None or episode != previous_episode:
            next_advantage = 0.0
        continuation = block_discounts[reverse_position] * float(gae_lambda)
        advantage = float(deltas[reverse_position]) + continuation * next_advantage
        block_advantages[start] = np.float32(advantage)
        block_returns[start] = np.float32(advantage + float(values[start]))
        next_advantage = advantage
        previous_episode = episode
    return block_advantages, block_returns, durations


def channel_marginal_distribution_entropy(action_probs: Any, candidate_masks: Any) -> Any:
    """Return normalized entropy of channel inclusion mass over a policy batch."""
    torch, _ = _torch_modules()
    masks = candidate_masks.float()
    if masks.ndim != 2 or action_probs.ndim != 2:
        raise ValueError("action_probs and candidate_masks must both be rank-2")
    if int(action_probs.shape[1]) != int(masks.shape[0]):
        raise ValueError("action_probs action dimension must match candidate_masks")
    n_sensors = int(masks.shape[1])
    if n_sensors <= 1:
        return torch.zeros((), device=action_probs.device, dtype=action_probs.dtype)
    inclusion_mass = (action_probs @ masks).mean(dim=0)
    total_mass = inclusion_mass.sum()
    if float(total_mass.detach().cpu().item()) <= 1.0e-12:
        return torch.zeros((), device=action_probs.device, dtype=action_probs.dtype)
    normalized = inclusion_mass / total_mass
    entropy = -(normalized * torch.log(normalized.clamp_min(1.0e-12))).sum()
    return entropy / float(np.log(float(n_sensors)))


def standardized_negative_cost_targets(costs: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Convert per-state feasible action costs to zero-mean value targets."""
    values = np.asarray(costs, dtype=np.float32)
    valid = np.asarray(feasible, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("costs and feasible must be matched two-dimensional arrays")
    targets = np.zeros_like(values, dtype=np.float32)
    for row in range(values.shape[0]):
        row_valid = valid[row]
        if not np.any(row_valid):
            raise ValueError("every row requires at least one finite feasible action")
        row_costs = values[row, row_valid]
        scale = max(float(np.std(row_costs)), 1.0e-6)
        targets[row, row_valid] = -(row_costs - float(np.mean(row_costs))) / scale
    return targets


def masked_soft_target_cross_entropy(
    actor_logits: Any,
    value_targets: Any,
    feasible: Any,
    row_mask: Any,
    *,
    temperature: float,
) -> Any:
    """Match feasible categorical preferences induced by forecast values."""
    torch, _ = _torch_modules()
    if actor_logits.ndim != 2 or value_targets.shape != actor_logits.shape:
        raise ValueError("actor logits and value targets must be matched rank-2 tensors")
    valid = feasible.bool()
    selected_rows = row_mask.bool()
    if valid.shape != actor_logits.shape or selected_rows.shape != actor_logits.shape[:1]:
        raise ValueError("feasible and row masks must match the actor batch")
    if not bool(selected_rows.any().detach().cpu().item()):
        return torch.zeros((), device=actor_logits.device, dtype=actor_logits.dtype)
    scale = max(float(temperature), 1.0e-6)
    target_logits = (value_targets / scale).masked_fill(~valid, -1.0e9)
    target_probs = torch.softmax(target_logits, dim=1)
    actor_log_probs = torch.log_softmax(actor_logits.masked_fill(~valid, -1.0e9), dim=1)
    return (-(target_probs * actor_log_probs).sum(dim=1))[selected_rows].mean()


@dataclass(frozen=True)
class CustomPPOConfig:
    total_timesteps: int = 100_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    channel_marginal_entropy_coef: float = 0.0
    vf_coef: float = 0.5
    awbc_coef: float = 0.1
    awbc_decay_timesteps: int = 0
    awbc_label_stride: int = 4
    awbc_event_only: bool = False
    bc_pretrain_steps: int = 0
    bc_pretrain_epochs: int = 4
    bc_pretrain_batch_size: int = 128
    bc_pretrain_loss_coef: float = 1.0
    bc_pretrain_target_mode: str = "hard"
    bc_pretrain_decision_only: bool = False
    bc_pretrain_forced_action_weight: float = 1.0
    bc_soft_temperature: float = 1.0
    forecast_value_aux_coef: float = 0.0
    forecast_value_aux_stride: int = 64
    forecast_value_aux_lookahead_steps: int = 0
    forecast_value_aux_loss: str = "mse"
    forecast_value_ranking_coef: float = 0.0
    forecast_value_aux_temperature: float = 1.0
    forecast_value_head_enabled: bool = False
    forecast_value_head_scale: float = 1.0
    forecast_value_head_hidden_dim: int = 128
    forecast_value_head_mode: str = "factorized"
    forecast_value_head_ignore_quality: bool = False
    forecast_value_trust_gate: bool = False
    forecast_value_trust_hidden_dim: int = 64
    subtype_aux_coef: float = 0.0
    subtype_aux_classes: int = 4
    subtype_aux_lookahead_steps: int = 0
    subtype_action_ce_coef: float = 0.0
    subtype_action_supervision_mode: str = "exact_action"
    subtype_action_event_only: bool = False
    subtype_action_margin_coef: float = 0.0
    subtype_action_margin: float = 0.5
    subtype_router_enabled: bool = False
    subtype_router_min_confidence: float = 0.0
    subtype_router_low_confidence_action: int = -1
    prior_kl_coef: float = 1.0
    embed_dim: int = 32
    hidden_dim: int = 128
    max_grad_norm: float = 0.5
    separate_actor_critic_grad_clip: bool = False
    greedy_lookahead_steps: int = 4
    awbc_teacher_mode: str = "oracle_greedy"
    awbc_teacher_event_lookahead_steps: int = 0
    awbc_teacher_alert_threshold: float = 0.5
    awbc_teacher_energy_mpc_horizon: int = 4
    awbc_teacher_energy_mpc_soc_bins: int = 16
    awbc_teacher_energy_mpc_low_soc_ratio: float = 0.25
    awbc_teacher_energy_mpc_high_soc_ratio: float = 0.75
    awbc_teacher_energy_mpc_terminal_soc_weight: float = 0.0
    awbc_teacher_energy_mpc_max_actions: int = 0
    awbc_teacher_energy_mpc_low_power_action: int = -1
    awbc_teacher_calm_action: int = -1
    awbc_teacher_event_action: int = -1
    awbc_teacher_calm_actions: tuple[int, ...] = ()
    awbc_teacher_event_actions: tuple[int, ...] = ()
    awbc_teacher_subtype_calm_action: int = -1
    awbc_teacher_subtype_particle_action: int = -1
    awbc_teacher_subtype_flux_action: int = -1
    awbc_teacher_subtype_thermal_action: int = -1
    awbc_teacher_dwell_steps: int = 1
    event_start_prob: float = 0.67
    use_action_mask: bool = True
    use_action_embedding: bool = True
    nonlinear_action_embedding: bool = False
    factorized_action_policy: bool = False
    trainable_action_prior: bool = True
    event_aware_critic: bool = True
    event_gated_actor: bool = False
    context_encoder_enabled: bool = False
    context_feature_dim: int = 0
    context_hidden_dim: int = 64
    context_fusion_mode: str = "concat"
    context_layer_norm: bool = False
    aligned_quality_action_score: bool = False
    quality_context_action_score: bool = False
    quality_context_pooling: str = "mean"
    onpolicy_action_value_coef: float = 0.0
    onpolicy_action_value_scale: float = 1.0
    candidate_interaction_score: bool = False
    direct_mask_action_score: bool = False
    direct_mask_action_primary: bool = False
    temporal_encoder_enabled: bool = False
    temporal_history_steps: int = 0
    temporal_state_dim: int = 0
    temporal_hidden_dim: int = 64
    decision_only_policy_updates: bool = False
    decision_block_credit: bool = False
    decision_block_reward_mode: str = "sum"
    soc_aux_horizon: int = 0
    soc_aux_coef: float = 0.0
    device: str = "auto"
    seed: int = 42
    history_path: str | None = None
    train_start_indices: tuple[int, ...] = ()
    train_start_min: int | None = None
    train_start_max: int | None = None


class SensorEmbedding:
    def __new__(cls, n_sensors: int, embed_dim: int) -> Any:
        _, nn = _torch_modules()

        class _SensorEmbedding(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(int(n_sensors), int(embed_dim))
                nn.init.normal_(self.embedding.weight, mean=0.0, std=0.08)

            def forward(self, sensor_indices: Any) -> Any:
                return self.embedding(sensor_indices)

        return _SensorEmbedding()


class ActionEmbedding:
    def __new__(cls, n_sensors: int, embed_dim: int, *, nonlinear: bool = False) -> Any:
        torch, nn = _torch_modules()

        class _ActionEmbedding(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sensor_embedding = SensorEmbedding(int(n_sensors), int(embed_dim))
                self.subset_encoder = (
                    nn.Sequential(
                        nn.Linear(int(embed_dim), int(embed_dim)),
                        nn.Tanh(),
                        nn.Linear(int(embed_dim), int(embed_dim)),
                        nn.Tanh(),
                    )
                    if bool(nonlinear)
                    else nn.Identity()
                )

            def forward(self, action_masks: Any) -> Any:
                masks = action_masks.float()
                sensor_ids = torch.arange(int(n_sensors), device=masks.device, dtype=torch.long)
                sensor_emb = self.sensor_embedding(sensor_ids)
                pooled = masks @ sensor_emb
                return self.subset_encoder(pooled)

        return _ActionEmbedding()


class MaskedActor:
    def __new__(
        cls,
        obs_dim: int,
        n_sensors: int,
        embed_dim: int,
        hidden_dim: int,
        n_actions: int | None = None,
        candidate_prior_logits: np.ndarray | None = None,
        use_action_embedding: bool = True,
        nonlinear_action_embedding: bool = False,
        factorized_action_policy: bool = False,
        trainable_action_prior: bool = True,
        event_gated: bool = False,
        subtype_aux_classes: int = 0,
        context_encoder_enabled: bool = False,
        context_feature_dim: int = 0,
        context_hidden_dim: int = 64,
        context_fusion_mode: str = "concat",
        context_layer_norm: bool = False,
        aligned_quality_action_score: bool = False,
        quality_context_action_score: bool = False,
        quality_context_pooling: str = "mean",
        onpolicy_action_value_enabled: bool = False,
        onpolicy_action_value_scale: float = 1.0,
        candidate_interaction_score: bool = False,
        direct_mask_action_score: bool = False,
        direct_mask_action_primary: bool = False,
        temporal_encoder_enabled: bool = False,
        temporal_history_steps: int = 0,
        temporal_state_dim: int = 0,
        temporal_hidden_dim: int = 64,
        forecast_value_head_enabled: bool = False,
        forecast_value_head_scale: float = 1.0,
        forecast_value_head_hidden_dim: int = 128,
        forecast_value_head_mode: str = "factorized",
        forecast_value_head_ignore_quality: bool = False,
        forecast_value_trust_gate: bool = False,
        forecast_value_trust_hidden_dim: int = 64,
        candidate_cost_features: np.ndarray | None = None,
    ) -> Any:
        torch, nn = _torch_modules()

        class _MaskedActor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_action_embedding = bool(use_action_embedding)
                self.factorized_action_policy = bool(factorized_action_policy)
                self.event_gated = bool(event_gated)
                self.n_actions = int(n_actions or 0)
                self.subtype_aux_classes = max(0, int(subtype_aux_classes))
                self.context_fusion_mode = str(context_fusion_mode)
                self.forecast_value_head_enabled = bool(forecast_value_head_enabled)
                self.forecast_value_head_scale = float(forecast_value_head_scale)
                self.forecast_value_head_mode = str(forecast_value_head_mode)
                self.forecast_value_head_ignore_quality = bool(
                    forecast_value_head_ignore_quality
                )
                self.forecast_value_trust_gate_enabled = bool(forecast_value_trust_gate)
                if self.forecast_value_head_mode not in {"factorized", "independent", "mask_structured"}:
                    raise ValueError(
                        "forecast_value_head_mode must be factorized, independent, or mask_structured"
                    )
                self.context_feature_dim = (
                    max(0, min(int(context_feature_dim), int(obs_dim) - 1))
                    if bool(context_encoder_enabled)
                    else 0
                )
                main_obs_dim = int(obs_dim) - int(self.context_feature_dim)
                self.aligned_quality_action_score = bool(aligned_quality_action_score)
                self.quality_context_action_score = bool(quality_context_action_score)
                self.quality_context_pooling = str(quality_context_pooling)
                if self.quality_context_pooling not in {"mean", "sum"}:
                    raise ValueError("quality_context_pooling must be mean or sum")
                self.onpolicy_action_value_enabled = bool(onpolicy_action_value_enabled)
                self.onpolicy_action_value_scale = float(onpolicy_action_value_scale)
                if (
                    self.aligned_quality_action_score or self.quality_context_action_score
                ) and self.context_feature_dim < int(n_sensors):
                    raise ValueError(
                        "quality action scoring requires one leading context feature per sensor"
                    )
                if self.quality_context_action_score and self.context_feature_dim <= int(n_sensors):
                    raise ValueError(
                        "quality-context action scoring requires context features after channel quality"
                    )
                self.quality_action_scale_raw = (
                    nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                    if self.aligned_quality_action_score
                    else None
                )
                self.quality_context_encoder = (
                    nn.Sequential(
                        nn.Linear(self.context_feature_dim - int(n_sensors), int(context_hidden_dim)),
                        nn.GELU(),
                        nn.Linear(int(context_hidden_dim), int(n_sensors)),
                        nn.Softplus(),
                    )
                    if self.quality_context_action_score
                    else None
                )
                self.quality_context_scale_raw = (
                    nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                    if self.quality_context_action_score
                    else None
                )
                self.candidate_interaction_head = (
                    nn.Sequential(
                        nn.Linear(2 * int(embed_dim), int(hidden_dim)),
                        nn.GELU(),
                        nn.Linear(int(hidden_dim), 1),
                    )
                    if bool(candidate_interaction_score)
                    else None
                )
                self.direct_mask_action_head = (
                    nn.Sequential(
                        nn.Linear(int(embed_dim) + int(n_sensors), int(hidden_dim)),
                        nn.GELU(),
                        nn.Linear(int(hidden_dim), 1),
                    )
                    if bool(direct_mask_action_score)
                    else None
                )
                self.direct_mask_action_primary = bool(direct_mask_action_primary)
                if self.direct_mask_action_primary and self.direct_mask_action_head is None:
                    raise ValueError(
                        "direct_mask_action_primary requires direct_mask_action_score"
                    )
                self.onpolicy_action_value_head = (
                    nn.Sequential(
                        nn.Linear(2 * int(embed_dim), int(hidden_dim)),
                        nn.GELU(),
                        nn.Linear(int(hidden_dim), 1),
                    )
                    if self.onpolicy_action_value_enabled
                    else None
                )
                self.temporal_history_steps = (
                    max(0, int(temporal_history_steps))
                    if bool(temporal_encoder_enabled)
                    else 0
                )
                self.temporal_state_dim = (
                    max(0, int(temporal_state_dim))
                    if bool(temporal_encoder_enabled)
                    else 0
                )
                self.temporal_flat_dim = (
                    2 * self.temporal_history_steps * self.temporal_state_dim
                )
                if bool(temporal_encoder_enabled) and (
                    self.temporal_history_steps <= 0
                    or self.temporal_state_dim <= 0
                    or self.temporal_flat_dim > main_obs_dim
                ):
                    raise ValueError(
                        "temporal encoder requires valid history/state dimensions within the main observation"
                    )
                if self.use_action_embedding:
                    self.action_embedding = ActionEmbedding(
                        int(n_sensors),
                        int(embed_dim),
                        nonlinear=bool(nonlinear_action_embedding),
                    )
                else:
                    if self.n_actions <= 0:
                        raise ValueError("n_actions must be provided when use_action_embedding=False")
                    self.action_embedding = nn.Embedding(self.n_actions, int(embed_dim))
                    nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.08)
                if self.temporal_flat_dim > 0:
                    self.temporal_encoder = nn.GRU(
                        input_size=2 * self.temporal_state_dim,
                        hidden_size=max(1, int(temporal_hidden_dim)),
                        batch_first=True,
                    )
                    remainder_dim = int(main_obs_dim) - int(self.temporal_flat_dim)
                    self.runtime_encoder = (
                        nn.Sequential(
                            nn.Linear(remainder_dim, int(embed_dim)),
                            nn.Tanh(),
                        )
                        if remainder_dim > 0
                        else None
                    )
                    fusion_dim = max(1, int(temporal_hidden_dim)) + (
                        int(embed_dim) if self.runtime_encoder is not None else 0
                    )
                    self.obs_encoder = nn.Sequential(
                        nn.Linear(fusion_dim, int(embed_dim)),
                        nn.Tanh(),
                    )
                else:
                    self.temporal_encoder = None
                    self.runtime_encoder = None
                    self.obs_encoder = nn.Sequential(
                        nn.Linear(int(main_obs_dim), int(hidden_dim)),
                        nn.Tanh(),
                        nn.Linear(int(hidden_dim), int(embed_dim)),
                        nn.Tanh(),
                    )
                self.context_encoder = (
                    nn.Sequential(
                        nn.Linear(int(self.context_feature_dim), int(context_hidden_dim)),
                        nn.Tanh(),
                        nn.Linear(int(context_hidden_dim), int(embed_dim)),
                        nn.Tanh(),
                    )
                    if self.context_feature_dim > 0
                    else None
                )
                self.context_fusion = (
                    nn.Sequential(
                        nn.Linear(int(embed_dim) * 2, int(embed_dim)),
                        nn.Tanh(),
                    )
                    if self.context_feature_dim > 0 and self.context_fusion_mode == "concat"
                    else None
                )
                self.context_gate = (
                    nn.Sequential(
                        nn.Linear(int(self.context_feature_dim), int(context_hidden_dim)),
                        nn.Tanh(),
                        nn.Linear(int(context_hidden_dim), int(embed_dim)),
                        nn.Sigmoid(),
                    )
                    if self.context_feature_dim > 0 and self.context_fusion_mode == "gated_add"
                    else None
                )
                if self.context_feature_dim > 0 and self.context_fusion_mode == "subtype_moe":
                    if self.subtype_aux_classes <= 1:
                        raise ValueError("subtype_moe requires at least two subtype auxiliary classes")
                    self.context_router = nn.Linear(int(embed_dim), self.subtype_aux_classes)
                    self.context_experts = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(int(embed_dim), int(embed_dim)),
                                nn.Tanh(),
                            )
                            for _ in range(self.subtype_aux_classes)
                        ]
                    )
                else:
                    self.context_router = None
                    self.context_experts = None
                self.context_norm = (
                    nn.LayerNorm(int(embed_dim))
                    if self.context_feature_dim > 0 and bool(context_layer_norm)
                    else None
                )
                if self.event_gated:
                    self.event_obs_encoder = nn.Sequential(
                        nn.Linear(int(main_obs_dim), int(hidden_dim)),
                        nn.Tanh(),
                        nn.Linear(int(hidden_dim), int(embed_dim)),
                        nn.Tanh(),
                    )
                    self.event_gate_alpha = nn.Parameter(torch.tensor(6.0, dtype=torch.float32))
                    self.event_gate_beta = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
                self.subtype_head = (
                    nn.Sequential(
                        nn.Linear(int(embed_dim), int(hidden_dim)),
                        nn.Tanh(),
                        nn.Linear(int(hidden_dim), self.subtype_aux_classes),
                    )
                    if self.subtype_aux_classes > 0 and self.context_router is None
                    else None
                )
                self.action_bias = nn.Linear(int(embed_dim), 1)
                self.factorized_action_head = (
                    nn.Linear(int(embed_dim), int(n_sensors))
                    if self.factorized_action_policy
                    else None
                )
                if self.forecast_value_head_enabled:
                    forecast_hidden = max(1, int(forecast_value_head_hidden_dim))
                    if self.forecast_value_head_mode == "independent":
                        if self.n_actions <= 0:
                            raise ValueError("independent forecast-value head requires n_actions")
                        self.forecast_state_encoder = nn.Sequential(
                            nn.Linear(int(obs_dim), forecast_hidden),
                            nn.GELU(),
                            nn.Linear(forecast_hidden, self.n_actions),
                        )
                        self.forecast_action_embedding = None
                        self.forecast_action_bias = None
                        self.forecast_mask_cost_regressor = None
                        self.register_buffer(
                            "forecast_candidate_cost_features",
                            torch.empty((0, 2), dtype=torch.float32),
                        )
                    elif self.forecast_value_head_mode == "mask_structured":
                        if self.context_feature_dim <= int(n_sensors):
                            raise ValueError(
                                "mask-structured forecast head requires quality plus alert context features"
                            )
                        features = np.asarray(candidate_cost_features, dtype=np.float32)
                        if features.shape != (self.n_actions, 2):
                            raise ValueError(
                                "mask-structured forecast head requires [n_actions, 2] cost features"
                            )
                        self.forecast_state_encoder = None
                        self.forecast_action_embedding = None
                        self.forecast_action_bias = None
                        self.forecast_mask_cost_regressor = MaskCostRegressor(
                            MaskCostRegressorConfig(
                                context_dim=(
                                    self.context_feature_dim - int(n_sensors)
                                    if self.forecast_value_head_ignore_quality
                                    else self.context_feature_dim
                                ),
                                sensor_count=int(n_sensors),
                                context_hidden_dim=forecast_hidden,
                                action_hidden_dim=forecast_hidden,
                                quality_feature_count=(
                                    0
                                    if self.forecast_value_head_ignore_quality
                                    else int(n_sensors)
                                ),
                            )
                        )
                        self.register_buffer(
                            "forecast_candidate_cost_features",
                            torch.as_tensor(features, dtype=torch.float32),
                        )
                    else:
                        self.forecast_state_encoder = nn.Sequential(
                            nn.Linear(int(obs_dim), forecast_hidden),
                            nn.GELU(),
                            nn.Linear(forecast_hidden, int(embed_dim)),
                            nn.Tanh(),
                        )
                        self.forecast_action_embedding = ActionEmbedding(
                            int(n_sensors),
                            int(embed_dim),
                            nonlinear=True,
                        )
                        self.forecast_action_bias = nn.Linear(int(embed_dim), 1)
                        self.forecast_mask_cost_regressor = None
                        self.register_buffer(
                            "forecast_candidate_cost_features",
                            torch.empty((0, 2), dtype=torch.float32),
                        )
                else:
                    self.forecast_state_encoder = None
                    self.forecast_action_embedding = None
                    self.forecast_action_bias = None
                    self.forecast_mask_cost_regressor = None
                    self.register_buffer(
                        "forecast_candidate_cost_features",
                        torch.empty((0, 2), dtype=torch.float32),
                    )
                self.forecast_value_trust_gate = (
                    nn.Sequential(
                        nn.Linear(int(embed_dim), max(1, int(forecast_value_trust_hidden_dim))),
                        nn.GELU(),
                        nn.Linear(max(1, int(forecast_value_trust_hidden_dim)), 1),
                        nn.Sigmoid(),
                    )
                    if self.forecast_value_head_enabled and self.forecast_value_trust_gate_enabled
                    else None
                )
                prior_len = int(n_actions or 0)
                if candidate_prior_logits is not None:
                    prior = torch.as_tensor(np.asarray(candidate_prior_logits, dtype=np.float32).reshape(-1))
                    prior_len = int(prior.numel())
                else:
                    prior = torch.zeros(prior_len, dtype=torch.float32)
                self.action_prior = (
                    nn.Parameter(prior, requires_grad=True)
                    if prior_len > 0 and bool(trainable_action_prior)
                    else None
                )

            def _split_obs(self, obs: Any) -> tuple[Any, Any | None]:
                if self.context_feature_dim <= 0:
                    return obs, None
                return obs[:, : -int(self.context_feature_dim)], obs[:, -int(self.context_feature_dim) :]

            def _action_embeddings(self, candidate_masks: Any) -> Any:
                if self.use_action_embedding:
                    return self.action_embedding(candidate_masks)
                action_ids = torch.arange(self.n_actions, device=candidate_masks.device, dtype=torch.long)
                return self.action_embedding(action_ids)

            def _encode_main_observation(self, main_obs: Any) -> Any:
                if self.temporal_encoder is None:
                    return self.obs_encoder(main_obs)
                history_size = self.temporal_history_steps * self.temporal_state_dim
                values = main_obs[:, :history_size].reshape(
                    main_obs.shape[0], self.temporal_history_steps, self.temporal_state_dim
                )
                masks = main_obs[:, history_size : 2 * history_size].reshape(
                    main_obs.shape[0], self.temporal_history_steps, self.temporal_state_dim
                )
                sequence = torch_concat([values, masks], dim=2)
                _, hidden = self.temporal_encoder(sequence)
                pieces = [hidden[-1]]
                if self.runtime_encoder is not None:
                    pieces.append(self.runtime_encoder(main_obs[:, 2 * history_size :]))
                return self.obs_encoder(torch_concat(pieces, dim=1))

            def encode_context(
                self,
                obs: Any,
                event_flag: Any | None = None,
            ) -> Any:
                main_obs, context_obs = self._split_obs(obs)
                context = self._encode_main_observation(main_obs)
                if self.context_encoder is not None and context_obs is not None:
                    context_extra = self.context_encoder(context_obs)
                    if self.context_fusion_mode == "subtype_moe" and self.context_router is not None:
                        router_probs = torch.softmax(self.context_router(context_extra), dim=1)
                        expert_values = torch.stack(
                            [expert(context_extra) for expert in self.context_experts],
                            dim=1,
                        )
                        context = context + torch.sum(router_probs.unsqueeze(-1) * expert_values, dim=1)
                    elif self.context_fusion_mode == "gated_add" and self.context_gate is not None:
                        gate = self.context_gate(context_obs)
                        context = context + gate * context_extra
                    elif self.context_fusion is not None:
                        context = self.context_fusion(torch_concat([context, context_extra], dim=1))
                    else:
                        context = context + context_extra
                    if self.context_norm is not None:
                        context = self.context_norm(context)
                if self.event_gated:
                    if event_flag is None:
                        event_flag = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
                    event = event_flag.float().reshape(obs.shape[0], 1)
                    gate = torch.sigmoid(self.event_gate_alpha * event + self.event_gate_beta)
                    event_context = self.event_obs_encoder(main_obs)
                    context = (1.0 - gate) * context + gate * event_context
                return context

            def subtype_logits(self, obs: Any, event_flag: Any | None = None) -> Any | None:
                if self.context_router is not None and self.context_encoder is not None:
                    _, context_obs = self._split_obs(obs)
                    if context_obs is None:
                        return None
                    return self.context_router(self.context_encoder(context_obs))
                if self.subtype_head is None:
                    return None
                return self.subtype_head(self.encode_context(obs, event_flag))

            def logits(
                self,
                obs: Any,
                candidate_masks: Any,
                action_mask: Any | None = None,
                event_flag: Any | None = None,
            ) -> Any:
                context = self.encode_context(obs, event_flag)
                if self.factorized_action_head is not None:
                    channel_logits = self.factorized_action_head(context)
                    masks = candidate_masks.float()
                    log_active = torch.nn.functional.logsigmoid(channel_logits)
                    log_inactive = torch.nn.functional.logsigmoid(-channel_logits)
                    logits = (
                        masks.unsqueeze(0) * log_active.unsqueeze(1)
                        + (1.0 - masks.unsqueeze(0)) * log_inactive.unsqueeze(1)
                    ).sum(dim=2)
                else:
                    action_emb = self._action_embeddings(candidate_masks)
                    logits = context @ action_emb.transpose(0, 1) / max(float(action_emb.shape[-1]) ** 0.5, 1.0)
                    logits = logits + self.action_bias(action_emb).reshape(1, -1)
                if self.factorized_action_head is None and self.candidate_interaction_head is not None:
                    state_actions = context.unsqueeze(1).expand(-1, action_emb.shape[0], -1)
                    candidate_actions = action_emb.unsqueeze(0).expand(context.shape[0], -1, -1)
                    interaction = self.candidate_interaction_head(
                        torch_concat([state_actions, candidate_actions], dim=2)
                    ).squeeze(-1)
                    logits = logits + interaction
                if self.factorized_action_head is None and self.direct_mask_action_head is not None:
                    state_actions = context.unsqueeze(1).expand(-1, candidate_masks.shape[0], -1)
                    raw_candidates = candidate_masks.float().unsqueeze(0).expand(context.shape[0], -1, -1)
                    direct_score = self.direct_mask_action_head(
                        torch_concat([state_actions, raw_candidates], dim=2)
                    ).squeeze(-1)
                    logits = direct_score if self.direct_mask_action_primary else logits + direct_score
                if self.factorized_action_head is None and self.quality_action_scale_raw is not None:
                    _, context_obs = self._split_obs(obs)
                    quality = context_obs[:, : int(n_sensors)]
                    masks = candidate_masks.float()
                    selected_count = masks.sum(dim=1).clamp_min(1.0)
                    selected_quality = quality @ masks.transpose(0, 1) / selected_count.reshape(1, -1)
                    scale = torch.nn.functional.softplus(self.quality_action_scale_raw)
                    logits = logits + scale * selected_quality
                if self.factorized_action_head is None and self.quality_context_encoder is not None:
                    _, context_obs = self._split_obs(obs)
                    quality = context_obs[:, : int(n_sensors)].clamp(0.0, 1.0)
                    alert_context = context_obs[:, int(n_sensors) :]
                    channel_utility = quality * self.quality_context_encoder(alert_context)
                    masks = candidate_masks.float()
                    candidate_utility = channel_utility @ masks.transpose(0, 1)
                    if self.quality_context_pooling == "mean":
                        selected_count = masks.sum(dim=1).clamp_min(1.0)
                        candidate_utility = candidate_utility / selected_count.reshape(1, -1)
                    scale = torch.nn.functional.softplus(self.quality_context_scale_raw)
                    logits = logits + scale * candidate_utility
                if self.factorized_action_head is None and self.action_prior is not None:
                    logits = logits + self.action_prior.reshape(1, -1)
                if self.factorized_action_head is None and self.forecast_value_head_enabled:
                    forecast_logits = self.forecast_value_logits(
                        obs,
                        candidate_masks,
                        action_mask,
                    )
                    if self.forecast_value_trust_gate is not None:
                        trust = self.forecast_value_trust_gate(context)
                        logits = logits + self.forecast_value_head_scale * trust * forecast_logits.detach()
                    else:
                        logits = logits + self.forecast_value_head_scale * forecast_logits.detach()
                if self.factorized_action_head is None and self.onpolicy_action_value_head is not None:
                    # Returns have a different scale from policy logits.  Normalize the
                    # candidate values per state before using them as a detached scorer.
                    action_values = self.onpolicy_action_values(
                        obs, candidate_masks, context=context
                    ).detach()
                    action_values = action_values - action_values.mean(dim=1, keepdim=True)
                    action_values = action_values / action_values.std(
                        dim=1, keepdim=True, unbiased=False
                    ).clamp_min(1.0e-6)
                    logits = logits + self.onpolicy_action_value_scale * action_values
                if action_mask is not None:
                    valid = action_mask.bool()
                    if valid.ndim == 1:
                        valid = valid.reshape(1, -1).expand_as(logits)
                    logits = logits.masked_fill(~valid, -1.0e9)
                return logits

            def onpolicy_action_values(self, obs: Any, candidate_masks: Any, *, context: Any | None = None) -> Any:
                if self.onpolicy_action_value_head is None:
                    raise RuntimeError("on-policy action-value head is not enabled")
                state = self.encode_context(obs) if context is None else context
                action_emb = self._action_embeddings(candidate_masks)
                state_actions = state.unsqueeze(1).expand(-1, action_emb.shape[0], -1)
                candidate_actions = action_emb.unsqueeze(0).expand(state.shape[0], -1, -1)
                return self.onpolicy_action_value_head(
                    torch_concat([state_actions, candidate_actions], dim=2)
                ).squeeze(-1)

            def forecast_value_logits(
                self,
                obs: Any,
                candidate_masks: Any,
                action_mask: Any | None = None,
            ) -> Any:
                if self.forecast_value_head_mode == "mask_structured":
                    if self.forecast_mask_cost_regressor is None:
                        raise RuntimeError("mask-structured forecast-value head is not enabled")
                    _, context_obs = self._split_obs(obs)
                    if context_obs is None:
                        raise RuntimeError("mask-structured forecast-value head requires context")
                    forecast_context = (
                        context_obs[:, int(candidate_masks.shape[1]) :]
                        if self.forecast_value_head_ignore_quality
                        else context_obs
                    )
                    logits = -self.forecast_mask_cost_regressor(
                        forecast_context,
                        candidate_masks,
                        self.forecast_candidate_cost_features,
                    )
                elif self.forecast_state_encoder is None:
                    raise RuntimeError("forecast-value head is not enabled")
                else:
                    state = self.forecast_state_encoder(obs)
                    if self.forecast_value_head_mode == "independent":
                        logits = state
                    else:
                        action_emb = self.forecast_action_embedding(candidate_masks)
                        logits = state @ action_emb.transpose(0, 1) / max(
                            float(action_emb.shape[-1]) ** 0.5,
                            1.0,
                        )
                        logits = logits + self.forecast_action_bias(action_emb).reshape(1, -1)
                if action_mask is not None:
                    valid = action_mask.bool()
                    if valid.ndim == 1:
                        valid = valid.reshape(1, -1).expand_as(logits)
                    logits = logits.masked_fill(~valid, -1.0e9)
                return logits

            def dist(
                self,
                obs: Any,
                candidate_masks: Any,
                action_mask: Any | None = None,
                event_flag: Any | None = None,
            ) -> Any:
                logits = self.logits(obs, candidate_masks, action_mask, event_flag)
                return torch.distributions.Categorical(logits=logits)

        return _MaskedActor()


class EventAwareCritic:
    def __new__(cls, obs_dim: int, hidden_dim: int, *, event_aware: bool = True) -> Any:
        _, nn = _torch_modules()

        class _EventAwareCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.event_aware = bool(event_aware)
                input_dim = int(obs_dim) + (1 if self.event_aware else 0)
                self.net = nn.Sequential(
                    nn.Linear(input_dim, int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), 1),
                )

            def forward(self, obs: Any, event_flag: Any) -> Any:
                if self.event_aware:
                    event = event_flag.float().reshape(obs.shape[0], 1)
                    x = torch_concat([obs, event], dim=1)
                else:
                    x = obs
                return self.net(x).reshape(-1)

        return _EventAwareCritic()


class ActorCritic:
    def __new__(
        cls,
        obs_dim: int,
        n_sensors: int,
        embed_dim: int,
        hidden_dim: int,
        n_actions: int | None = None,
        candidate_prior_logits: np.ndarray | None = None,
        use_action_embedding: bool = True,
        nonlinear_action_embedding: bool = False,
        factorized_action_policy: bool = False,
        trainable_action_prior: bool = True,
        event_aware_critic: bool = True,
        event_gated_actor: bool = False,
        soc_aux_horizon: int = 0,
        subtype_aux_classes: int = 0,
        context_encoder_enabled: bool = False,
        context_feature_dim: int = 0,
        context_hidden_dim: int = 64,
        context_fusion_mode: str = "concat",
        context_layer_norm: bool = False,
        aligned_quality_action_score: bool = False,
        quality_context_action_score: bool = False,
        quality_context_pooling: str = "mean",
        onpolicy_action_value_enabled: bool = False,
        onpolicy_action_value_scale: float = 1.0,
        candidate_interaction_score: bool = False,
        direct_mask_action_score: bool = False,
        direct_mask_action_primary: bool = False,
        temporal_encoder_enabled: bool = False,
        temporal_history_steps: int = 0,
        temporal_state_dim: int = 0,
        temporal_hidden_dim: int = 64,
        forecast_value_head_enabled: bool = False,
        forecast_value_head_scale: float = 1.0,
        forecast_value_head_hidden_dim: int = 128,
        forecast_value_head_mode: str = "factorized",
        forecast_value_head_ignore_quality: bool = False,
        forecast_value_trust_gate: bool = False,
        forecast_value_trust_hidden_dim: int = 64,
        candidate_cost_features: np.ndarray | None = None,
    ) -> Any:
        _, nn = _torch_modules()

        class _ActorCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.actor = MaskedActor(
                    int(obs_dim),
                    int(n_sensors),
                    int(embed_dim),
                    int(hidden_dim),
                    n_actions=n_actions,
                    candidate_prior_logits=candidate_prior_logits,
                    use_action_embedding=bool(use_action_embedding),
                    nonlinear_action_embedding=bool(nonlinear_action_embedding),
                    factorized_action_policy=bool(factorized_action_policy),
                    trainable_action_prior=bool(trainable_action_prior),
                    event_gated=bool(event_gated_actor),
                    subtype_aux_classes=int(subtype_aux_classes),
                    context_encoder_enabled=bool(context_encoder_enabled),
                    context_feature_dim=int(context_feature_dim),
                    context_hidden_dim=int(context_hidden_dim),
                    context_fusion_mode=str(context_fusion_mode),
                    context_layer_norm=bool(context_layer_norm),
                    aligned_quality_action_score=bool(aligned_quality_action_score),
                    quality_context_action_score=bool(quality_context_action_score),
                    quality_context_pooling=str(quality_context_pooling),
                    onpolicy_action_value_enabled=bool(onpolicy_action_value_enabled),
                    onpolicy_action_value_scale=float(onpolicy_action_value_scale),
                    candidate_interaction_score=bool(candidate_interaction_score),
                    direct_mask_action_score=bool(direct_mask_action_score),
                    direct_mask_action_primary=bool(direct_mask_action_primary),
                    temporal_encoder_enabled=bool(temporal_encoder_enabled),
                    temporal_history_steps=int(temporal_history_steps),
                    temporal_state_dim=int(temporal_state_dim),
                    temporal_hidden_dim=int(temporal_hidden_dim),
                    forecast_value_head_enabled=bool(forecast_value_head_enabled),
                    forecast_value_head_scale=float(forecast_value_head_scale),
                    forecast_value_head_hidden_dim=int(forecast_value_head_hidden_dim),
                    forecast_value_head_mode=str(forecast_value_head_mode),
                    forecast_value_head_ignore_quality=bool(
                        forecast_value_head_ignore_quality
                    ),
                    forecast_value_trust_gate=bool(forecast_value_trust_gate),
                    forecast_value_trust_hidden_dim=int(forecast_value_trust_hidden_dim),
                    candidate_cost_features=candidate_cost_features,
                )
                self.critic = EventAwareCritic(int(obs_dim), int(hidden_dim), event_aware=bool(event_aware_critic))
                self.soc_aux_horizon = max(0, int(soc_aux_horizon))
                self.soc_aux_head = (
                    nn.Sequential(
                        nn.Linear(int(obs_dim) + 1, int(hidden_dim)),
                        nn.Tanh(),
                        nn.Linear(int(hidden_dim), self.soc_aux_horizon),
                    )
                    if self.soc_aux_horizon > 0
                    else None
                )

            def value(self, obs: Any, event_flag: Any) -> Any:
                return self.critic(obs, event_flag)

            def predict_soc(self, obs: Any, event_flag: Any) -> Any | None:
                if self.soc_aux_head is None:
                    return None
                event = event_flag.float().reshape(obs.shape[0], 1)
                return self.soc_aux_head(torch_concat([obs, event], dim=1))

            def subtype_logits(self, obs: Any, event_flag: Any) -> Any | None:
                return self.actor.subtype_logits(obs, event_flag)

            def dist(
                self,
                obs: Any,
                candidate_masks: Any,
                action_mask: Any | None = None,
                event_flag: Any | None = None,
            ) -> Any:
                return self.actor.dist(obs, candidate_masks, action_mask, event_flag)

        return _ActorCritic()


class CustomPPO:
    def __init__(
        self,
        *,
        truth_df: pd.DataFrame,
        sensor_specs: list[SensorSpecV2],
        constraints: PowerConstraintsV2,
        env_cfg: WarmupEnvConfig,
        oracle: LinearFrozenForecastOracle,
        candidate_masks: np.ndarray,
        cfg: CustomPPOConfig,
        candidate_prior_logits: np.ndarray | None = None,
        training_scenarios: list[tuple[pd.DataFrame, WarmupEnvConfig, object]] | None = None,
    ) -> None:
        torch, _ = _torch_modules()
        self.truth_df = truth_df
        self.sensor_specs = list(sensor_specs)
        self.constraints = constraints
        self.env_cfg = env_cfg
        self.oracle = oracle
        self.training_scenarios = list(
            training_scenarios or [(self.truth_df, self.env_cfg, self.oracle)]
        )
        if not self.training_scenarios:
            raise ValueError("training_scenarios must contain at least one scene")
        self._training_scenario_cursor = 0
        self.candidate_masks_np = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(sensor_specs))
        if self.candidate_masks_np.shape[0] == 0:
            raise ValueError("candidate_masks must contain at least one action")
        self.cfg = cfg
        self.candidate_prior_logits_np = (
            None
            if candidate_prior_logits is None
            else np.asarray(candidate_prior_logits, dtype=np.float32).reshape(self.candidate_masks_np.shape[0])
        )
        self.device = _select_device(torch, str(cfg.device))
        self.rng = np.random.default_rng(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))

        probe_env = self._make_env(seed_offset=0)
        probe_obs, _ = probe_env.reset()
        self._training_scenario_cursor = 0
        self.obs_dim = int(np.asarray(probe_obs).shape[0])
        self.context_feature_dim = int(cfg.context_feature_dim)
        if self.context_feature_dim <= 0:
            self.context_feature_dim = int(getattr(probe_env, "alert_context_feature_dim", 0))
        if bool(cfg.context_encoder_enabled) and self.context_feature_dim <= 0:
            raise ValueError("context_encoder_enabled requires alert/context features in the environment state")
        if self.context_feature_dim > 0:
            self.cfg = replace(self.cfg, context_feature_dim=int(self.context_feature_dim))
        steady_budget = max(float(self.constraints.per_step_budget or 1.0), 1.0e-9)
        peak_budget = max(float(self.constraints.startup_peak_budget or steady_budget), 1.0e-9)
        candidate_cost_features = np.asarray(
            [
                [
                    sum(float(self.sensor_specs[idx].power_cost) for idx in np.flatnonzero(mask))
                    / steady_budget,
                    sum(float(self.sensor_specs[idx].startup_peak_power) for idx in np.flatnonzero(mask))
                    / peak_budget,
                ]
                for mask in self.candidate_masks_np
            ],
            dtype=np.float32,
        )
        self.model = ActorCritic(
            obs_dim=self.obs_dim,
            n_sensors=len(self.sensor_specs),
            embed_dim=int(cfg.embed_dim),
            hidden_dim=int(cfg.hidden_dim),
            n_actions=int(self.candidate_masks_np.shape[0]),
            candidate_prior_logits=self.candidate_prior_logits_np,
            use_action_embedding=bool(cfg.use_action_embedding),
            nonlinear_action_embedding=bool(cfg.nonlinear_action_embedding),
            factorized_action_policy=bool(cfg.factorized_action_policy),
            trainable_action_prior=bool(cfg.trainable_action_prior),
            event_aware_critic=bool(cfg.event_aware_critic),
            event_gated_actor=bool(cfg.event_gated_actor),
            soc_aux_horizon=int(cfg.soc_aux_horizon),
            subtype_aux_classes=(
                int(cfg.subtype_aux_classes)
                if float(cfg.subtype_aux_coef) > 0.0 and "event_subtype_id" in self.truth_df.columns
                else 0
            ),
            context_encoder_enabled=bool(cfg.context_encoder_enabled),
            context_feature_dim=int(self.context_feature_dim),
            context_hidden_dim=int(cfg.context_hidden_dim),
            context_fusion_mode=str(cfg.context_fusion_mode),
            context_layer_norm=bool(cfg.context_layer_norm),
            aligned_quality_action_score=bool(cfg.aligned_quality_action_score),
            quality_context_action_score=bool(cfg.quality_context_action_score),
            quality_context_pooling=str(cfg.quality_context_pooling),
            onpolicy_action_value_enabled=float(cfg.onpolicy_action_value_coef) > 0.0,
            onpolicy_action_value_scale=float(cfg.onpolicy_action_value_scale),
            candidate_interaction_score=bool(cfg.candidate_interaction_score),
            direct_mask_action_score=bool(cfg.direct_mask_action_score),
            direct_mask_action_primary=bool(cfg.direct_mask_action_primary),
            temporal_encoder_enabled=bool(cfg.temporal_encoder_enabled),
            temporal_history_steps=int(self.env_cfg.lookback),
            temporal_state_dim=len(self.env_cfg.state_columns),
            temporal_hidden_dim=int(cfg.temporal_hidden_dim),
            forecast_value_head_enabled=bool(cfg.forecast_value_head_enabled),
            forecast_value_head_scale=float(cfg.forecast_value_head_scale),
            forecast_value_head_hidden_dim=int(cfg.forecast_value_head_hidden_dim),
            forecast_value_head_mode=str(cfg.forecast_value_head_mode),
            forecast_value_head_ignore_quality=bool(
                cfg.forecast_value_head_ignore_quality
            ),
            forecast_value_trust_gate=bool(cfg.forecast_value_trust_gate),
            forecast_value_trust_hidden_dim=int(cfg.forecast_value_trust_hidden_dim),
            candidate_cost_features=candidate_cost_features,
        ).to(self.device)
        self.candidate_masks_t = torch_tensor(self.candidate_masks_np.astype(np.float32), device=self.device)
        self.candidate_prior_logits_t = (
            None
            if self.candidate_prior_logits_np is None
            else torch_tensor(self.candidate_prior_logits_np.astype(np.float32), device=self.device)
        )
        self.candidate_power_np = np.asarray(
            [
                float(
                    sum(
                        float(self.sensor_specs[idx].power_cost)
                        for idx in np.flatnonzero(mask)
                    )
                )
                for mask in self.candidate_masks_np
            ],
            dtype=np.float32,
        )
        self._energy_mpc_teacher_cache: dict[tuple[Any, ...], int] = {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg.learning_rate))
        self.history: list[dict[str, float | int]] = []

    def _clip_update_gradients(self) -> None:
        _, nn = _torch_modules()
        max_norm = float(self.cfg.max_grad_norm)
        if not bool(self.cfg.separate_actor_critic_grad_clip):
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            return
        nn.utils.clip_grad_norm_(self.model.actor.parameters(), max_norm)
        nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm)
        if self.model.soc_aux_head is not None:
            nn.utils.clip_grad_norm_(self.model.soc_aux_head.parameters(), max_norm)

    def train(
        self,
        on_update: Callable[["CustomPPO", int, int, dict[str, float | int]], None] | None = None,
    ) -> "CustomPPO":
        bc_steps = max(0, int(self.cfg.bc_pretrain_steps))
        if bc_steps > 0:
            metrics = self.bc_pretrain(bc_steps)
            metrics["timesteps"] = 0
            self.history.append(metrics)
            self._flush_history()
            print(
                "custom_ppo_bc_pretrain "
                f"steps={bc_steps} "
                f"loss={float(metrics['loss']):.6f} "
                f"accuracy={float(metrics.get('bc_accuracy', float('nan'))):.3f} "
                f"subtype_acc={float(metrics.get('subtype_aux_accuracy', float('nan'))):.3f} "
                f"unique_actions={int(metrics.get('greedy_unique_actions', 0))}",
                flush=True,
            )
            if on_update is not None:
                on_update(self, 0, 0, metrics)
        steps_done = 0
        for update_idx, rollout_steps in enumerate(
            full_rollout_schedule(self.cfg.total_timesteps, self.cfg.n_steps),
            start=1,
        ):
            self._active_awbc_coef = self._effective_awbc_coef(steps_done)
            batch = self.collect_rollout(int(rollout_steps), seed_offset=(update_idx - 1) * 997)
            metrics = self.update(batch)
            steps_done += int(batch["obs"].shape[0])
            metrics["timesteps"] = int(steps_done)
            metrics["rollout_reward_mean"] = float(np.mean(batch["rewards"]))
            metrics["rollout_reward_std"] = float(np.std(batch["rewards"]))
            metrics["rollout_return_mean"] = float(np.mean(batch["returns"]))
            metrics["rollout_return_std"] = float(np.std(batch["returns"]))
            self.history.append(metrics)
            self._flush_history()
            if on_update is not None:
                on_update(self, int(update_idx), int(steps_done), metrics)
            print(
                "custom_ppo_update "
                f"update={update_idx} timesteps={steps_done} "
                f"loss={float(metrics['loss']):.6f} "
                f"entropy={float(metrics['entropy']):.6f} "
                f"adv_std={float(metrics['advantage_std']):.6f} "
                f"soc_aux={float(metrics.get('soc_aux_loss', float('nan'))):.6f} "
                f"subtype_aux={float(metrics.get('subtype_aux_loss', float('nan'))):.6f} "
                f"subtype_acc={float(metrics.get('subtype_aux_accuracy', float('nan'))):.3f} "
                f"subtype_action_ce={float(metrics.get('subtype_action_ce_loss', float('nan'))):.6f} "
                f"subtype_action_margin={float(metrics.get('subtype_action_margin_loss', float('nan'))):.6f} "
                f"forecast_value_aux={float(metrics.get('forecast_value_aux_loss', float('nan'))):.6f} "
                f"forecast_value_rate={float(metrics.get('forecast_value_label_rate', 0.0)):.3f} "
                f"awbc_coef={float(metrics['awbc_coef']):.6f} "
                f"awbc_label_rate={float(metrics['awbc_label_rate']):.3f}",
                flush=True,
            )
        return self

    def collect_rollout(self, n_steps: int, *, seed_offset: int = 0) -> dict[str, np.ndarray]:
        torch, _ = _torch_modules()
        env = self._make_env(seed_offset=seed_offset)
        env.reset(
            start_idx=self._sample_start_idx(
                self._sampling_window_steps(int(n_steps)), seed_offset=seed_offset, env=env
            )
        )

        obs_rows: list[np.ndarray] = []
        action_rows: list[int] = []
        greedy_rows: list[int] = []
        logprob_rows: list[float] = []
        reward_rows: list[float] = []
        done_rows: list[float] = []
        value_rows: list[float] = []
        event_rows: list[float] = []
        action_mask_rows: list[np.ndarray] = []
        decision_rows: list[float] = []
        awbc_valid_rows: list[float] = []
        subtype_label_rows: list[int] = []
        subtype_valid_rows: list[float] = []
        forecast_value_target_rows: list[np.ndarray] = []
        forecast_value_valid_rows: list[float] = []
        soc_rows: list[float] = []
        episode_rows: list[int] = []

        last_done = False
        episode_id = 0
        label_stride = max(1, int(self.cfg.awbc_label_stride))
        for step in range(int(n_steps)):
            obs_np = env._state().astype(np.float32)
            action_mask_np = (
                feasible_candidate_mask(env, self.candidate_masks_np)
                if bool(self.cfg.use_action_mask)
                else np.ones(self.candidate_masks_np.shape[0], dtype=bool)
            )
            subtype_label, subtype_valid = self._subtype_aux_label(env)
            event_flag = float(env.online_event_context())
            obs_t = torch_tensor(obs_np.reshape(1, -1), device=self.device)
            action_mask_t = torch_tensor(action_mask_np.reshape(1, -1), device=self.device, dtype=torch.bool)
            event_t = torch_tensor(np.asarray([event_flag], dtype=np.float32), device=self.device)
            with torch.no_grad():
                dist = self.model.dist(obs_t, self.candidate_masks_t, action_mask_t, event_t)
                action_t = dist.sample()
                logprob_t = dist.log_prob(action_t)
                value_t = self.model.value(obs_t, event_t)
            action = int(action_t.detach().cpu().item())
            decision_available = (
                int(getattr(env, "elapsed_steps", 0)) <= 0
                or int(getattr(env, "dwell_hold_remaining", 0)) <= 0
            )
            forecast_value_valid = (
                float(self.cfg.forecast_value_aux_coef) > 0.0
                and step % max(1, int(self.cfg.forecast_value_aux_stride)) == 0
                and decision_available
            )
            if forecast_value_valid:
                forecast_costs = oracle_greedy_candidate_costs(
                    env,
                    self.candidate_masks_np,
                    lookahead_steps=max(
                        1,
                        int(self.cfg.forecast_value_aux_lookahead_steps)
                        or int(self.cfg.greedy_lookahead_steps),
                    ),
                )
                forecast_targets = standardized_negative_cost_targets(
                    forecast_costs.reshape(1, -1),
                    action_mask_np.reshape(1, -1),
                )[0]
            else:
                forecast_targets = np.zeros(len(action_mask_np), dtype=np.float32)
            should_label = (
                float(self._current_awbc_coef()) > 0.0
                and (step % label_stride == 0)
                and self._awbc_label_allowed(subtype_label, subtype_valid)
            )
            if should_label:
                greedy = self._awbc_teacher_action(env, action_mask_np)
                awbc_valid = 1.0
            else:
                greedy = action
                awbc_valid = 0.0
            block_gain = 0.0
            block_cost = 0.0
            if str(self.env_cfg.reward_proxy_mode) in {
                "forecast_block_gain",
                "forecast_block_relative_gain",
            } and decision_available:
                block_gain = forecast_block_gain(
                    env,
                    self.candidate_masks_np[action],
                    np.asarray(env.previous_action_mask, dtype=bool),
                    horizon=max(1, int(self.cfg.forecast_value_aux_lookahead_steps) or int(self.cfg.greedy_lookahead_steps)),
                    relative=(str(self.env_cfg.reward_proxy_mode) == "forecast_block_relative_gain"),
                )
            if str(self.env_cfg.reward_proxy_mode) == "forecast_block_absolute" and decision_available:
                block_cost = forecast_block_cost(
                    env,
                    self.candidate_masks_np[action],
                    horizon=max(
                        1,
                        int(self.cfg.forecast_value_aux_lookahead_steps)
                        or int(self.cfg.greedy_lookahead_steps),
                    ),
                )
            _, reward, done, info = env.step_mask(self.candidate_masks_np[action])
            if str(self.env_cfg.reward_proxy_mode) in {
                "forecast_block_gain",
                "forecast_block_relative_gain",
            }:
                reward = float(block_gain) - float(info.get("shaping_penalty", 0.0))
            elif str(self.env_cfg.reward_proxy_mode) == "forecast_block_absolute":
                reward = -float(block_cost) - float(info.get("shaping_penalty", 0.0))

            obs_rows.append(obs_np)
            action_rows.append(action)
            greedy_rows.append(int(greedy))
            logprob_rows.append(float(logprob_t.detach().cpu().item()))
            reward_rows.append(float(reward))
            done_rows.append(float(done))
            value_rows.append(float(value_t.detach().cpu().item()))
            event_rows.append(event_flag)
            action_mask_rows.append(action_mask_np.astype(bool))
            # A policy transition is available only when the environment can
            # accept a new mask. The number of feasible candidates is not a
            # proxy for dwell expiry: forced hold steps may still expose many
            # feasible candidates while disallowing a new decision.
            decision_rows.append(float(decision_available))
            awbc_valid_rows.append(float(awbc_valid))
            subtype_label_rows.append(int(subtype_label))
            subtype_valid_rows.append(float(subtype_valid))
            forecast_value_target_rows.append(np.asarray(forecast_targets, dtype=np.float32))
            forecast_value_valid_rows.append(float(forecast_value_valid))
            soc_rows.append(float(info.get("soc_ratio", 1.0)))
            episode_rows.append(int(episode_id))
            last_done = bool(done)
            if done and step < int(n_steps) - 1:
                env = self._make_env(seed_offset=seed_offset + step + 1)
                env.reset(
                    start_idx=self._sample_start_idx(
                        self._sampling_window_steps(int(n_steps)),
                        seed_offset=seed_offset + step + 1,
                        env=env,
                    )
                )
                last_done = False
                episode_id += 1

        if last_done:
            last_value = 0.0
        else:
            obs_t = torch_tensor(env._state().astype(np.float32).reshape(1, -1), device=self.device)
            event_t = torch_tensor(
                np.asarray([float(env.online_event_context())], dtype=np.float32),
                device=self.device,
            )
            with torch.no_grad():
                last_value = float(self.model.value(obs_t, event_t).detach().cpu().item())

        rewards = np.asarray(reward_rows, dtype=np.float32)
        dones = np.asarray(done_rows, dtype=np.float32)
        values = np.asarray(value_rows, dtype=np.float32)
        advantages = compute_gae(
            rewards,
            values,
            dones,
            last_value=last_value,
            gamma=float(self.cfg.gamma),
            gae_lambda=float(self.cfg.gae_lambda),
        )
        returns = advantages + values
        if bool(self.cfg.decision_block_credit):
            policy_advantages, policy_returns, decision_durations = compute_decision_block_credit(
                rewards,
                values,
                dones,
                np.asarray(decision_rows, dtype=np.float32),
                np.asarray(episode_rows, dtype=np.int64),
                gamma=float(self.cfg.gamma),
                gae_lambda=float(self.cfg.gae_lambda),
                reward_mode=str(self.cfg.decision_block_reward_mode),
            )
        else:
            policy_advantages = advantages.astype(np.float32)
            policy_returns = returns.astype(np.float32)
            decision_durations = np.zeros(len(rewards), dtype=np.int64)
        soc_targets, soc_mask = build_future_soc_targets(
            np.asarray(soc_rows, dtype=np.float32),
            np.asarray(episode_rows, dtype=np.int64),
            horizon=int(self.cfg.soc_aux_horizon),
        )
        return {
            "obs": np.vstack(obs_rows).astype(np.float32),
            "actions": np.asarray(action_rows, dtype=np.int64),
            "greedy_actions": np.asarray(greedy_rows, dtype=np.int64),
            "old_logprobs": np.asarray(logprob_rows, dtype=np.float32),
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "advantages": advantages.astype(np.float32),
            "returns": returns.astype(np.float32),
            "policy_advantages": policy_advantages.astype(np.float32),
            "policy_returns": policy_returns.astype(np.float32),
            "decision_durations": decision_durations.astype(np.int64),
            "event_flags": np.asarray(event_rows, dtype=np.float32),
            "action_masks": np.vstack(action_mask_rows).astype(bool),
            "decision_rows": np.asarray(decision_rows, dtype=np.float32),
            "awbc_valid": np.asarray(awbc_valid_rows, dtype=np.float32),
            "subtype_labels": np.asarray(subtype_label_rows, dtype=np.int64),
            "subtype_valid": np.asarray(subtype_valid_rows, dtype=np.float32),
            "forecast_value_targets": np.vstack(forecast_value_target_rows).astype(np.float32),
            "forecast_value_valid": np.asarray(forecast_value_valid_rows, dtype=np.float32),
            "soc_aux_targets": soc_targets.astype(np.float32),
            "soc_aux_mask": soc_mask.astype(np.float32),
        }

    def collect_teacher_batch(self, n_steps: int, *, seed_offset: int = 0) -> dict[str, np.ndarray]:
        env = self._make_env(seed_offset=seed_offset)
        env.reset(
            start_idx=self._sample_start_idx(
                self._sampling_window_steps(int(n_steps)), seed_offset=seed_offset, env=env
            )
        )

        obs_rows: list[np.ndarray] = []
        teacher_rows: list[int] = []
        event_rows: list[float] = []
        action_mask_rows: list[np.ndarray] = []
        subtype_label_rows: list[int] = []
        subtype_valid_rows: list[float] = []
        teacher_distribution_rows: list[np.ndarray] = []
        teacher_cost_rows: list[np.ndarray] = []
        decision_rows: list[float] = []
        episode_id = 0
        for step in range(int(n_steps)):
            obs_np = env._state().astype(np.float32)
            action_mask_np = (
                feasible_candidate_mask(env, self.candidate_masks_np)
                if bool(self.cfg.use_action_mask)
                else np.ones(self.candidate_masks_np.shape[0], dtype=bool)
            )
            if str(self.cfg.bc_pretrain_target_mode) in {
                "hard_forecast_value",
                "soft_forecast_value",
                "forecast_value_regression",
            }:
                teacher_costs = oracle_greedy_candidate_costs(
                    env,
                    self.candidate_masks_np,
                    lookahead_steps=int(self.cfg.greedy_lookahead_steps),
                )
                feasible = np.flatnonzero(action_mask_np)
                if feasible.size:
                    teacher = int(feasible[np.argmin(teacher_costs[feasible])])
                else:
                    teacher = int(np.argmin(teacher_costs))
                if str(self.cfg.bc_pretrain_target_mode) == "soft_forecast_value":
                    teacher_distribution = soft_forecast_value_targets(
                        teacher_costs.reshape(1, -1),
                        action_mask_np.reshape(1, -1),
                        temperature=float(self.cfg.bc_soft_temperature),
                    )[0]
                elif str(self.cfg.bc_pretrain_target_mode) == "hard_forecast_value":
                    teacher_distribution = np.zeros(len(action_mask_np), dtype=np.float32)
                    if 0 <= teacher < len(action_mask_np):
                        teacher_distribution[teacher] = 1.0
                else:
                    teacher_distribution = np.zeros(len(action_mask_np), dtype=np.float32)
            else:
                teacher_costs = np.full(len(action_mask_np), np.nan, dtype=np.float32)
                teacher = int(self._awbc_teacher_action(env, action_mask_np))
                teacher_distribution = np.zeros(len(action_mask_np), dtype=np.float32)
                if 0 <= teacher < len(action_mask_np):
                    teacher_distribution[teacher] = 1.0
            if not (0 <= teacher < len(action_mask_np) and bool(action_mask_np[teacher])):
                feasible = np.flatnonzero(action_mask_np)
                teacher = int(feasible[0]) if feasible.size else 0
            subtype_label, subtype_valid = self._subtype_aux_label(env)

            obs_rows.append(obs_np)
            teacher_rows.append(int(teacher))
            teacher_distribution_rows.append(teacher_distribution.astype(np.float32))
            teacher_cost_rows.append(np.asarray(teacher_costs, dtype=np.float32))
            event_rows.append(float(env.online_event_context()))
            action_mask_rows.append(action_mask_np.astype(bool))
            subtype_label_rows.append(int(subtype_label))
            subtype_valid_rows.append(float(subtype_valid))
            decision_available = (
                int(getattr(env, "elapsed_steps", 0)) <= 0
                or int(getattr(env, "dwell_hold_remaining", 0)) <= 0
            )
            decision_rows.append(float(decision_available))

            _, _, done, _ = env.step_mask(self.candidate_masks_np[teacher])
            if done and step < int(n_steps) - 1:
                env = self._make_env(seed_offset=seed_offset + step + 1)
                env.reset(
                    start_idx=self._sample_start_idx(
                        self._sampling_window_steps(int(n_steps)),
                        seed_offset=seed_offset + step + 1,
                        env=env,
                    )
                )
                episode_id += 1

        return {
            "obs": np.vstack(obs_rows).astype(np.float32),
            "teacher_actions": np.asarray(teacher_rows, dtype=np.int64),
            "teacher_distributions": np.vstack(teacher_distribution_rows).astype(np.float32),
            "teacher_costs": np.vstack(teacher_cost_rows).astype(np.float32),
            "event_flags": np.asarray(event_rows, dtype=np.float32),
            "action_masks": np.vstack(action_mask_rows).astype(bool),
            "subtype_labels": np.asarray(subtype_label_rows, dtype=np.int64),
            "subtype_valid": np.asarray(subtype_valid_rows, dtype=np.float32),
            "decision_rows": np.asarray(decision_rows, dtype=np.float32),
            "episode_ids": np.full(len(teacher_rows), int(episode_id), dtype=np.int64),
        }

    def bc_pretrain(self, n_steps: int) -> dict[str, float | int]:
        torch, nn = _torch_modules()
        batch = self.collect_teacher_batch(int(n_steps), seed_offset=91_000)
        if bool(self.cfg.bc_pretrain_decision_only):
            decision_mask = np.asarray(batch["decision_rows"], dtype=np.float32) > 0.5
            if not np.any(decision_mask):
                raise RuntimeError("decision-only BC pretraining produced no executable decision rows")
            for key in (
                "obs",
                "teacher_actions",
                "teacher_distributions",
                "teacher_costs",
                "event_flags",
                "action_masks",
                "subtype_labels",
                "subtype_valid",
                "decision_rows",
            ):
                batch[key] = batch[key][decision_mask]
        obs = torch_tensor(batch["obs"], device=self.device)
        teacher_actions = torch_tensor(batch["teacher_actions"], device=self.device, dtype=torch.long)
        teacher_distributions = torch_tensor(batch["teacher_distributions"], device=self.device)
        teacher_cost_targets = torch_tensor(
            standardized_negative_cost_targets(batch["teacher_costs"], batch["action_masks"])
            if str(self.cfg.bc_pretrain_target_mode) == "forecast_value_regression"
            else np.zeros_like(batch["teacher_costs"], dtype=np.float32),
            device=self.device,
        )
        event_flags = torch_tensor(batch["event_flags"], device=self.device)
        action_masks = torch_tensor(batch["action_masks"], device=self.device, dtype=torch.bool)
        subtype_labels = torch_tensor(batch["subtype_labels"], device=self.device, dtype=torch.long)
        subtype_valid = torch_tensor(batch["subtype_valid"], device=self.device)

        n = int(obs.shape[0])
        batch_size = max(1, min(int(self.cfg.bc_pretrain_batch_size), n))
        indices = np.arange(n)
        rows: list[dict[str, float]] = []
        for _ in range(max(1, int(self.cfg.bc_pretrain_epochs))):
            self.rng.shuffle(indices)
            for start in range(0, n, batch_size):
                idx = torch_tensor(indices[start : start + batch_size], device=self.device, dtype=torch.long)
                mb_obs = obs[idx]
                mb_teacher = teacher_actions[idx]
                mb_teacher_distribution = teacher_distributions[idx]
                mb_teacher_cost_targets = teacher_cost_targets[idx]
                mb_events = event_flags[idx]
                mb_masks = action_masks[idx]
                mb_subtype_labels = subtype_labels[idx]
                mb_subtype_valid = subtype_valid[idx]
                dist = self.model.dist(mb_obs, self.candidate_masks_t, mb_masks, mb_events)
                decision_weight = float(np.clip(self.cfg.bc_pretrain_forced_action_weight, 0.0, 1.0))
                row_weights = torch.where(
                    torch.as_tensor(batch["decision_rows"], device=self.device, dtype=torch.float32)[idx] > 0.5,
                    torch.ones_like(mb_teacher, dtype=torch.float32),
                    torch.full_like(mb_teacher, decision_weight, dtype=torch.float32),
                )
                if str(self.cfg.bc_pretrain_target_mode) == "soft_forecast_value":
                    per_row_bc_loss = -(mb_teacher_distribution * torch.log(dist.probs.clamp_min(1.0e-12))).sum(dim=1)
                    bc_loss = (per_row_bc_loss * row_weights).sum() / row_weights.sum().clamp_min(1.0e-6)
                elif str(self.cfg.bc_pretrain_target_mode) == "forecast_value_regression":
                    actor_logits = (
                        self.model.actor.forecast_value_logits(
                            mb_obs,
                            self.candidate_masks_t,
                            mb_masks,
                        )
                        if bool(self.cfg.forecast_value_head_enabled)
                        else self.model.actor.logits(
                            mb_obs,
                            self.candidate_masks_t,
                            mb_masks,
                            mb_events,
                        )
                    )
                    valid_logits = torch.where(mb_masks, actor_logits, mb_teacher_cost_targets)
                    if str(self.cfg.forecast_value_aux_loss).strip().lower() == "smooth_l1":
                        per_row_regression_loss = nn.functional.smooth_l1_loss(
                            valid_logits[mb_masks],
                            mb_teacher_cost_targets[mb_masks],
                            reduction="none",
                        )
                        per_row_regression_loss = torch.zeros_like(valid_logits).masked_scatter(
                            mb_masks, per_row_regression_loss
                        ).sum(dim=1)
                    else:
                        squared_error = (valid_logits - mb_teacher_cost_targets).square()
                        per_row_regression_loss = squared_error.masked_fill(~mb_masks, 0.0).sum(dim=1)
                    best_actions = torch.argmax(
                        mb_teacher_cost_targets.masked_fill(~mb_masks, -1.0e9),
                        dim=1,
                    )
                    per_row_ranking_loss = nn.functional.cross_entropy(
                        actor_logits.masked_fill(~mb_masks, -1.0e9),
                        best_actions,
                        reduction="none",
                    )
                    per_row_bc_loss = per_row_regression_loss + float(self.cfg.forecast_value_ranking_coef) * per_row_ranking_loss
                    bc_loss = (per_row_bc_loss * row_weights).sum() / row_weights.sum().clamp_min(1.0e-6)
                else:
                    per_row_bc_loss = -dist.log_prob(mb_teacher)
                    bc_loss = (per_row_bc_loss * row_weights).sum() / row_weights.sum().clamp_min(1.0e-6)
                subtype_aux_loss, subtype_aux_acc = self._subtype_aux_loss(
                    mb_obs,
                    mb_events,
                    mb_subtype_labels,
                    mb_subtype_valid,
                )
                loss = float(self.cfg.bc_pretrain_loss_coef) * bc_loss + float(self.cfg.subtype_aux_coef) * subtype_aux_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._clip_update_gradients()
                self.optimizer.step()
                pred = torch.argmax(dist.probs, dim=1)
                rows.append(
                    {
                        "loss": float(loss.detach().cpu().item()),
                        "bc_loss": float(bc_loss.detach().cpu().item()),
                        "subtype_aux_loss": float(subtype_aux_loss.detach().cpu().item()),
                        "subtype_aux_accuracy": float(subtype_aux_acc),
                        "entropy": float(dist.entropy().mean().detach().cpu().item()),
                        "accuracy": float((pred == mb_teacher).float().mean().detach().cpu().item()),
                    }
                )

        return {
            "loss": _mean_metric(rows, "loss"),
            "policy_loss": float("nan"),
            "value_loss": float("nan"),
            "entropy": _mean_metric(rows, "entropy"),
            "awbc_loss": _mean_metric(rows, "bc_loss"),
            "prior_kl_loss": float("nan"),
            "soc_aux_loss": float("nan"),
            "subtype_aux_loss": _mean_metric(rows, "subtype_aux_loss"),
            "subtype_aux_accuracy": _mean_metric(rows, "subtype_aux_accuracy"),
            "advantage_mean": float("nan"),
            "advantage_std": float("nan"),
            "event_rate": float(np.mean(batch["event_flags"])),
            "awbc_label_rate": 1.0,
            "greedy_unique_actions": int(np.unique(batch["teacher_actions"]).size),
            "bc_pretrain": 1,
            "bc_steps": int(n),
            "bc_decision_rows": int(n),
            "bc_accuracy": _mean_metric(rows, "accuracy"),
        }

    def _subtype_aux_label(self, env: WarmupSchedulingEnv) -> tuple[int, float]:
        if float(self.cfg.subtype_aux_coef) <= 0.0 or "event_subtype_id" not in self.truth_df.columns:
            return 0, 0.0
        subtype_ids = self.truth_df["event_subtype_id"].to_numpy(dtype=int)
        lookahead_end = min(
            len(subtype_ids),
            int(env.current_idx) + max(0, int(self.cfg.subtype_aux_lookahead_steps)) + 1,
        )
        window = subtype_ids[int(env.current_idx) : lookahead_end]
        active = window[window > 0]
        label = int(active[0]) if active.size else 0
        label = int(np.clip(label, 0, max(0, int(self.cfg.subtype_aux_classes) - 1)))
        return label, 1.0

    def _awbc_label_allowed(self, subtype_label: int, subtype_valid: float) -> bool:
        if not bool(self.cfg.awbc_event_only):
            return True
        return float(subtype_valid) > 0.5 and int(subtype_label) > 0

    def _awbc_teacher_action(self, env: WarmupSchedulingEnv, action_mask: np.ndarray) -> int:
        mode = str(self.cfg.awbc_teacher_mode or "oracle_greedy")
        if mode == "event_pair":
            calm_idx = int(self.cfg.awbc_teacher_calm_action)
            event_idx = int(self.cfg.awbc_teacher_event_action)
            if calm_idx >= 0 and event_idx >= 0:
                lookahead_end = min(
                    len(env.event_flags),
                    int(env.current_idx) + max(0, int(self.cfg.awbc_teacher_event_lookahead_steps)) + 1,
                )
                trigger = bool(np.any(env.event_flags[int(env.current_idx) : lookahead_end]))
                idx = event_idx if trigger else calm_idx
                if 0 <= idx < len(action_mask) and bool(action_mask[idx]):
                    return int(idx)
        if mode == "event_cyclic":
            lookahead_end = min(
                len(env.event_flags),
                int(env.current_idx) + max(0, int(self.cfg.awbc_teacher_event_lookahead_steps)) + 1,
            )
            trigger = bool(np.any(env.event_flags[int(env.current_idx) : lookahead_end]))
            pool = (
                tuple(int(x) for x in self.cfg.awbc_teacher_event_actions)
                if trigger
                else tuple(int(x) for x in self.cfg.awbc_teacher_calm_actions)
            )
            if pool:
                dwell = max(1, int(self.cfg.awbc_teacher_dwell_steps))
                start_idx = int(getattr(env, "episode_start_idx", int(env.current_idx)))
                phase = max(0, int(env.current_idx) - start_idx) // dwell
                for offset in range(len(pool)):
                    idx = int(pool[(int(phase) + offset) % len(pool)])
                    if 0 <= idx < len(action_mask) and bool(action_mask[idx]):
                        return int(idx)
        if mode in {"subtype_auto", "subtype_static_auto"}:
            idx = self._subtype_preferred_action(env, action_mask)
            if 0 <= idx < len(action_mask) and bool(action_mask[idx]):
                return int(idx)
        if mode == "context_alert":
            labels = ("particle", "flux", "thermal")
            values = {
                label: float(self.truth_df.iloc[int(env.current_idx)].get(f"agent_context_{label}_alert", 0.0))
                for label in labels
            }
            label = max(values, key=values.get)
            subtype_id = labels.index(label) + 1 if values[label] >= float(self.cfg.awbc_teacher_alert_threshold) else 0
            actions = (
                int(self.cfg.awbc_teacher_subtype_calm_action),
                int(self.cfg.awbc_teacher_subtype_particle_action),
                int(self.cfg.awbc_teacher_subtype_flux_action),
                int(self.cfg.awbc_teacher_subtype_thermal_action),
            )
            idx = actions[int(subtype_id)]
            if 0 <= idx < len(action_mask) and bool(action_mask[idx]):
                return int(idx)
        if mode == "energy_mpc":
            return self._energy_mpc_teacher_action(env, action_mask)
        return oracle_greedy_candidate_index(
            env,
            self.candidate_masks_np,
            lookahead_steps=int(self.cfg.greedy_lookahead_steps),
        )

    def _subtype_preferred_action(self, env: WarmupSchedulingEnv, action_mask: np.ndarray) -> int:
        subtype_ids = (
            self.truth_df["event_subtype_id"].to_numpy(dtype=int)
            if "event_subtype_id" in self.truth_df.columns
            else np.zeros(len(self.truth_df), dtype=int)
        )
        lookahead_end = min(
            len(subtype_ids),
            int(env.current_idx) + max(0, int(self.cfg.awbc_teacher_event_lookahead_steps)) + 1,
        )
        window = subtype_ids[int(env.current_idx) : lookahead_end]
        active = window[window > 0]
        subtype_id = int(active[0]) if active.size else 0
        subtype_actions = {
            0: int(self.cfg.awbc_teacher_subtype_calm_action),
            1: int(self.cfg.awbc_teacher_subtype_particle_action),
            2: int(self.cfg.awbc_teacher_subtype_flux_action),
            3: int(self.cfg.awbc_teacher_subtype_thermal_action),
        }
        idx = int(subtype_actions.get(subtype_id, -1))
        if 0 <= idx < len(action_mask) and bool(action_mask[idx]):
            return int(idx)
        return -1

    def _energy_mpc_teacher_action(self, env: WarmupSchedulingEnv, action_mask: np.ndarray) -> int:
        action_mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        feasible = np.flatnonzero(action_mask)
        if feasible.size == 0:
            return 0
        cache_key = self._energy_mpc_cache_key(env)
        cached = self._energy_mpc_teacher_cache.get(cache_key)
        if cached is not None and 0 <= int(cached) < len(action_mask) and bool(action_mask[int(cached)]):
            return int(cached)

        snapshot = snapshot_env(env)
        best_idx = int(feasible[0])
        best_cost = float("inf")
        horizon = max(1, int(self.cfg.awbc_teacher_energy_mpc_horizon))
        for idx in self._energy_mpc_candidate_indices(feasible, env, action_mask):
            restore_env(env, snapshot)
            total_cost = 0.0
            discount = 1.0
            done = False
            _, reward, done, _ = env.step_mask(self.candidate_masks_np[int(idx)])
            step_cost = -float(reward)
            total_cost += step_cost if np.isfinite(step_cost) else float("inf")
            for _ in range(1, horizon):
                if done:
                    break
                tail_mask = (
                    feasible_candidate_mask(env, self.candidate_masks_np)
                    if bool(self.cfg.use_action_mask)
                    else np.ones(self.candidate_masks_np.shape[0], dtype=bool)
                )
                tail_idx = self._energy_mpc_tail_action(env, tail_mask)
                discount *= float(self.cfg.gamma)
                _, reward, done, _ = env.step_mask(self.candidate_masks_np[int(tail_idx)])
                step_cost = -float(reward)
                total_cost += discount * (step_cost if np.isfinite(step_cost) else float("inf"))
            total_cost += self._energy_mpc_terminal_soc_penalty(env)
            if total_cost < best_cost:
                best_cost = float(total_cost)
                best_idx = int(idx)
        restore_env(env, snapshot)
        self._energy_mpc_teacher_cache[cache_key] = int(best_idx)
        return int(best_idx)

    def _energy_mpc_candidate_indices(
        self,
        feasible: np.ndarray,
        env: WarmupSchedulingEnv,
        action_mask: np.ndarray,
    ) -> list[int]:
        feasible = np.asarray(feasible, dtype=int).reshape(-1)
        max_actions = max(0, int(self.cfg.awbc_teacher_energy_mpc_max_actions))
        if max_actions <= 0 or feasible.size <= max_actions:
            return [int(x) for x in feasible.tolist()]

        keep: set[int] = set()
        subtype_idx = self._subtype_preferred_action(env, action_mask)
        if 0 <= subtype_idx < len(action_mask) and bool(action_mask[subtype_idx]):
            keep.add(int(subtype_idx))
        low_idx = self._low_power_feasible_action(action_mask)
        if 0 <= low_idx < len(action_mask) and bool(action_mask[low_idx]):
            keep.add(int(low_idx))
        explicit_low = int(self.cfg.awbc_teacher_energy_mpc_low_power_action)
        if 0 <= explicit_low < len(action_mask) and bool(action_mask[explicit_low]):
            keep.add(int(explicit_low))

        if self.candidate_prior_logits_np is not None:
            scores = self.candidate_prior_logits_np[feasible]
        else:
            scores = -self.candidate_power_np[feasible]
        order = np.argsort(scores)[::-1]
        for item in order:
            keep.add(int(feasible[int(item)]))
            if len(keep) >= max_actions:
                break
        return [int(x) for x in feasible.tolist() if int(x) in keep]

    def _energy_mpc_tail_action(self, env: WarmupSchedulingEnv, action_mask: np.ndarray) -> int:
        action_mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        soc_ratio = self._env_soc_ratio(env)
        low_soc = float(self.cfg.awbc_teacher_energy_mpc_low_soc_ratio)
        if soc_ratio <= low_soc:
            explicit = int(self.cfg.awbc_teacher_energy_mpc_low_power_action)
            if 0 <= explicit < len(action_mask) and bool(action_mask[explicit]):
                return int(explicit)
            return self._low_power_feasible_action(action_mask)

        subtype_idx = self._subtype_preferred_action(env, action_mask)
        if 0 <= subtype_idx < len(action_mask) and bool(action_mask[subtype_idx]):
            return int(subtype_idx)

        if self.candidate_prior_logits_np is not None:
            feasible = np.flatnonzero(action_mask)
            if feasible.size:
                return int(feasible[int(np.argmax(self.candidate_prior_logits_np[feasible]))])
        return self._low_power_feasible_action(action_mask)

    def _low_power_feasible_action(self, action_mask: np.ndarray) -> int:
        feasible = np.flatnonzero(np.asarray(action_mask, dtype=bool).reshape(-1))
        if feasible.size == 0:
            return 0
        powers = self.candidate_power_np[feasible]
        return int(feasible[int(np.argmin(powers))])

    def _energy_mpc_terminal_soc_penalty(self, env: WarmupSchedulingEnv) -> float:
        weight = max(0.0, float(self.cfg.awbc_teacher_energy_mpc_terminal_soc_weight))
        if weight <= 0.0 or not bool(self.env_cfg.energy_account_enabled):
            return 0.0
        capacity = max(float(self.env_cfg.energy_capacity), 1.0e-6)
        target = float(self.env_cfg.reserve_energy) + max(0.0, float(self.env_cfg.soc_soft_penalty_buffer))
        deficit_ratio = max(0.0, target - float(getattr(env, "current_energy", 0.0))) / capacity
        return float(weight * deficit_ratio * deficit_ratio)

    def _energy_mpc_cache_key(self, env: WarmupSchedulingEnv) -> tuple[Any, ...]:
        bins = max(1, int(self.cfg.awbc_teacher_energy_mpc_soc_bins))
        bucket = int(np.clip(round(self._env_soc_ratio(env) * float(bins - 1)), 0, bins - 1))
        previous = self._mask_int(np.asarray(getattr(env, "previous_action_mask"), dtype=bool))
        runtime = tuple(
            (int(runtime.mode), int(runtime.warm_remaining))
            for runtime in getattr(env, "runtimes").values()
        )
        return (int(env.current_idx), int(bucket), int(previous), runtime)

    @staticmethod
    def _env_soc_ratio(env: WarmupSchedulingEnv) -> float:
        try:
            return float(env._soc_ratio())
        except Exception:
            return 1.0

    @staticmethod
    def _mask_int(mask: np.ndarray) -> int:
        value = 0
        for idx, active in enumerate(np.asarray(mask, dtype=bool).reshape(-1)):
            if bool(active):
                value |= 1 << int(idx)
        return int(value)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float | int]:
        torch, nn = _torch_modules()
        obs = torch_tensor(batch["obs"], device=self.device)
        actions = torch_tensor(batch["actions"], device=self.device, dtype=torch.long)
        greedy_actions = torch_tensor(batch["greedy_actions"], device=self.device, dtype=torch.long)
        old_logprobs = torch_tensor(batch["old_logprobs"], device=self.device)
        returns = torch_tensor(batch["returns"], device=self.device)
        advantages = torch_tensor(batch["advantages"], device=self.device)
        policy_advantages = torch_tensor(
            batch.get("policy_advantages", batch["advantages"]), device=self.device
        )
        event_flags = torch_tensor(batch["event_flags"], device=self.device)
        action_masks = torch_tensor(batch["action_masks"], device=self.device, dtype=torch.bool)
        decision_rows = torch_tensor(
            batch.get("decision_rows", policy_decision_mask(batch["action_masks"])),
            device=self.device,
        )
        awbc_valid = torch_tensor(batch["awbc_valid"], device=self.device)
        subtype_labels = torch_tensor(batch["subtype_labels"], device=self.device, dtype=torch.long)
        subtype_valid = torch_tensor(batch["subtype_valid"], device=self.device)
        soc_aux_targets = torch_tensor(batch["soc_aux_targets"], device=self.device)
        soc_aux_mask = torch_tensor(batch["soc_aux_mask"], device=self.device)
        forecast_value_targets = torch_tensor(batch["forecast_value_targets"], device=self.device)
        forecast_value_valid = torch_tensor(batch["forecast_value_valid"], device=self.device)
        decision_policy = bool(self.cfg.decision_only_policy_updates or self.cfg.decision_block_credit)
        if policy_advantages.numel() > 1:
            norm_rows = decision_rows > 0.5 if decision_policy else torch.ones_like(decision_rows, dtype=torch.bool)
            if bool(norm_rows.any().detach().cpu().item()):
                norm_advantages = policy_advantages[norm_rows]
                policy_advantages = (policy_advantages - norm_advantages.mean()) / (norm_advantages.std(unbiased=False) + 1e-8)

        # Record pre-update policy confidence on genuine decision rows. These
        # diagnostics do not enter the loss and isolate learner credit issues.
        with torch.no_grad():
            diagnostic_dist = self.model.dist(obs, self.candidate_masks_t, action_masks, event_flags)
            diagnostic_rows = decision_rows > 0.5
            if bool(diagnostic_rows.any().detach().cpu().item()):
                diagnostic_probs = diagnostic_dist.probs[diagnostic_rows]
                diagnostic_actions = actions[diagnostic_rows]
                diagnostic_greedy = greedy_actions[diagnostic_rows]
                diagnostic_advantages = policy_advantages[diagnostic_rows]
                diagnostic_selected_prob = diagnostic_probs.gather(1, diagnostic_actions.reshape(-1, 1)).reshape(-1)
                diagnostic_greedy_prob = diagnostic_probs.gather(1, diagnostic_greedy.reshape(-1, 1)).reshape(-1)
                diagnostic_max_prob, diagnostic_argmax = diagnostic_probs.max(dim=1)
                decision_diagnostic_metrics = {
                    "decision_advantage_mean": float(diagnostic_advantages.mean().cpu().item()),
                    "decision_advantage_std": float(diagnostic_advantages.std(unbiased=False).cpu().item()),
                    "decision_selected_prob": float(diagnostic_selected_prob.mean().cpu().item()),
                    "decision_greedy_prob": float(diagnostic_greedy_prob.mean().cpu().item()),
                    "decision_max_prob": float(diagnostic_max_prob.mean().cpu().item()),
                    "decision_argmax_match_rate": float((diagnostic_argmax == diagnostic_actions).float().mean().cpu().item()),
                    "decision_greedy_match_rate": float((diagnostic_argmax == diagnostic_greedy).float().mean().cpu().item()),
                    "decision_entropy": float(diagnostic_dist.entropy()[diagnostic_rows].mean().cpu().item()),
                }
            else:
                decision_diagnostic_metrics = {name: float("nan") for name in (
                    "decision_advantage_mean", "decision_advantage_std",
                    "decision_selected_prob", "decision_greedy_prob",
                    "decision_max_prob", "decision_argmax_match_rate",
                    "decision_greedy_match_rate", "decision_entropy",
                )}

        n = int(obs.shape[0])
        batch_size = max(1, min(int(self.cfg.batch_size), n))
        indices = np.arange(n)
        loss_rows: list[dict[str, float]] = []
        for _ in range(int(self.cfg.n_epochs)):
            self.rng.shuffle(indices)
            for start in range(0, n, batch_size):
                idx = torch_tensor(indices[start : start + batch_size], device=self.device, dtype=torch.long)
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_greedy = greedy_actions[idx]
                mb_old_logprobs = old_logprobs[idx]
                mb_returns = returns[idx]
                mb_advantages = policy_advantages[idx]
                mb_events = event_flags[idx]
                mb_masks = action_masks[idx]
                mb_decision_rows = decision_rows[idx]
                mb_awbc_valid = awbc_valid[idx]
                mb_subtype_labels = subtype_labels[idx]
                mb_subtype_valid = subtype_valid[idx]
                mb_soc_targets = soc_aux_targets[idx]
                mb_soc_mask = soc_aux_mask[idx]
                mb_forecast_value_targets = forecast_value_targets[idx]
                mb_forecast_value_valid = forecast_value_valid[idx]

                dist = self.model.dist(mb_obs, self.candidate_masks_t, mb_masks, mb_events)
                new_logprobs = dist.log_prob(mb_actions)
                policy_weights = (
                    mb_decision_rows
                    if decision_policy
                    else torch.ones_like(mb_decision_rows)
                )
                policy_denominator = policy_weights.sum().clamp_min(1.0)
                entropy = (dist.entropy() * policy_weights).sum() / policy_denominator
                channel_marginal_entropy = channel_marginal_distribution_entropy(
                    dist.probs[policy_weights > 0.5],
                    self.candidate_masks_t,
                ) if bool((policy_weights > 0.5).any().detach().cpu().item()) else torch.zeros((), device=dist.probs.device)
                values = self.model.value(mb_obs, mb_events)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - float(self.cfg.clip_range),
                    1.0 + float(self.cfg.clip_range),
                )
                policy_terms = torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages)
                policy_loss = -(policy_terms * policy_weights).sum() / policy_denominator
                value_loss = nn.functional.mse_loss(values, mb_returns)
                onpolicy_action_value_loss = torch.zeros((), device=self.device)
                if float(self.cfg.onpolicy_action_value_coef) > 0.0:
                    action_values = self.model.actor.onpolicy_action_values(
                        mb_obs, self.candidate_masks_t
                    )
                    selected_values = action_values.gather(1, mb_actions.reshape(-1, 1)).reshape(-1)
                    onpolicy_action_value_loss = nn.functional.mse_loss(selected_values, mb_returns)
                awbc_loss = advantage_weighted_bc_loss(dist, mb_greedy, mb_advantages, mb_awbc_valid)
                prior_kl_loss = prior_kl_regularizer(
                    dist,
                    self.candidate_prior_logits_t,
                    mb_masks,
                )
                soc_aux_loss = torch.zeros((), device=self.device)
                if int(self.cfg.soc_aux_horizon) > 0 and float(self.cfg.soc_aux_coef) > 0.0:
                    soc_pred = self.model.predict_soc(mb_obs, mb_events)
                    if soc_pred is not None and float(mb_soc_mask.detach().sum().cpu().item()) > 0.0:
                        sq_error = (soc_pred - mb_soc_targets) ** 2
                        soc_aux_loss = (sq_error * mb_soc_mask).sum() / (mb_soc_mask.sum() + 1.0e-8)
                subtype_aux_loss, subtype_aux_acc = self._subtype_aux_loss(
                    mb_obs,
                    mb_events,
                    mb_subtype_labels,
                    mb_subtype_valid,
                )
                subtype_action_ce_loss, subtype_action_margin_loss, subtype_action_valid_rate = (
                    self._subtype_action_losses(dist, mb_masks, mb_subtype_labels, mb_subtype_valid)
                )
                forecast_value_aux_loss = torch.zeros((), device=self.device)
                forecast_rows = mb_forecast_value_valid > 0.5
                if (
                    float(self.cfg.forecast_value_aux_coef) > 0.0
                    and bool(forecast_rows.any().detach().cpu().item())
                ):
                    actor_logits = (
                        self.model.actor.forecast_value_logits(
                            mb_obs,
                            self.candidate_masks_t,
                            mb_masks,
                        )
                        if bool(self.cfg.forecast_value_head_enabled)
                        else self.model.actor.logits(
                            mb_obs,
                            self.candidate_masks_t,
                            mb_masks,
                            mb_events,
                        )
                    )
                    valid_entries = mb_masks & forecast_rows.unsqueeze(1)
                    if str(self.cfg.forecast_value_aux_loss).strip().lower() == "soft_ce":
                        forecast_value_aux_loss = masked_soft_target_cross_entropy(
                            actor_logits,
                            mb_forecast_value_targets,
                            mb_masks,
                            forecast_rows,
                            temperature=float(self.cfg.forecast_value_aux_temperature),
                        )
                    elif str(self.cfg.forecast_value_aux_loss).strip().lower() == "smooth_l1":
                        regression_loss = nn.functional.smooth_l1_loss(
                            actor_logits[valid_entries],
                            mb_forecast_value_targets[valid_entries],
                        )
                        best_actions = torch.argmax(
                            mb_forecast_value_targets.masked_fill(~mb_masks, -1.0e9),
                            dim=1,
                        )
                        ranking_loss = nn.functional.cross_entropy(
                            actor_logits[forecast_rows].masked_fill(
                                ~mb_masks[forecast_rows], -1.0e9
                            ),
                            best_actions[forecast_rows],
                        )
                        forecast_value_aux_loss = (
                            regression_loss
                            + float(self.cfg.forecast_value_ranking_coef) * ranking_loss
                        )
                    else:
                        squared_error = (actor_logits - mb_forecast_value_targets).square()
                        forecast_value_aux_loss = squared_error[valid_entries].mean()
                loss = (
                    policy_loss
                    + float(self.cfg.vf_coef) * value_loss
                    + float(self.cfg.onpolicy_action_value_coef) * onpolicy_action_value_loss
                    - float(self.cfg.ent_coef) * entropy
                    - float(self.cfg.channel_marginal_entropy_coef) * channel_marginal_entropy
                    + float(self._current_awbc_coef()) * awbc_loss
                    + float(self.cfg.prior_kl_coef) * prior_kl_loss
                    + float(self.cfg.soc_aux_coef) * soc_aux_loss
                    + float(self.cfg.subtype_aux_coef) * subtype_aux_loss
                    + float(self.cfg.subtype_action_ce_coef) * subtype_action_ce_loss
                    + float(self.cfg.subtype_action_margin_coef) * subtype_action_margin_loss
                    + float(self.cfg.forecast_value_aux_coef) * forecast_value_aux_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._clip_update_gradients()
                self.optimizer.step()
                loss_rows.append(
                    {
                        "loss": float(loss.detach().cpu().item()),
                        "policy_loss": float(policy_loss.detach().cpu().item()),
                        "value_loss": float(value_loss.detach().cpu().item()),
                        "onpolicy_action_value_loss": float(onpolicy_action_value_loss.detach().cpu().item()),
                        "entropy": float(entropy.detach().cpu().item()),
                        "channel_marginal_entropy": float(
                            channel_marginal_entropy.detach().cpu().item()
                        ),
                        "awbc_loss": float(awbc_loss.detach().cpu().item()),
                        "prior_kl_loss": float(prior_kl_loss.detach().cpu().item()),
                        "soc_aux_loss": float(soc_aux_loss.detach().cpu().item()),
                        "subtype_aux_loss": float(subtype_aux_loss.detach().cpu().item()),
                        "subtype_aux_accuracy": float(subtype_aux_acc),
                        "subtype_action_ce_loss": float(subtype_action_ce_loss.detach().cpu().item()),
                        "subtype_action_margin_loss": float(subtype_action_margin_loss.detach().cpu().item()),
                        "subtype_action_valid_rate": float(subtype_action_valid_rate),
                        "forecast_value_aux_loss": float(forecast_value_aux_loss.detach().cpu().item()),
                    }
                )

        return {
            "loss": _mean_metric(loss_rows, "loss"),
            "policy_loss": _mean_metric(loss_rows, "policy_loss"),
            "value_loss": _mean_metric(loss_rows, "value_loss"),
            "onpolicy_action_value_loss": _mean_metric(loss_rows, "onpolicy_action_value_loss"),
            "entropy": _mean_metric(loss_rows, "entropy"),
            "channel_marginal_entropy": _mean_metric(
                loss_rows, "channel_marginal_entropy"
            ),
            "awbc_loss": _mean_metric(loss_rows, "awbc_loss"),
            "prior_kl_loss": _mean_metric(loss_rows, "prior_kl_loss"),
            "soc_aux_loss": _mean_metric(loss_rows, "soc_aux_loss"),
            "subtype_aux_loss": _mean_metric(loss_rows, "subtype_aux_loss"),
            "subtype_aux_accuracy": _mean_metric(loss_rows, "subtype_aux_accuracy"),
            "subtype_action_ce_loss": _mean_metric(loss_rows, "subtype_action_ce_loss"),
            "subtype_action_margin_loss": _mean_metric(loss_rows, "subtype_action_margin_loss"),
            "subtype_action_valid_rate": _mean_metric(loss_rows, "subtype_action_valid_rate"),
            "forecast_value_aux_loss": _mean_metric(loss_rows, "forecast_value_aux_loss"),
            "forecast_value_label_rate": float(np.mean(batch["forecast_value_valid"])),
            "advantage_mean": float(np.mean(batch["advantages"])),
            "advantage_std": float(np.std(batch["advantages"])),
            "event_rate": float(np.mean(batch["event_flags"])),
            "awbc_label_rate": float(np.mean(batch["awbc_valid"])),
            "awbc_coef": float(self._current_awbc_coef()),
            "greedy_unique_actions": int(np.unique(batch["greedy_actions"]).size),
            "policy_decision_rate": float(np.mean(batch.get("decision_rows", policy_decision_mask(batch["action_masks"])) )),
            **decision_diagnostic_metrics,
        }

    def _effective_awbc_coef(self, timesteps: int) -> float:
        base = max(0.0, float(self.cfg.awbc_coef))
        decay_steps = max(0, int(self.cfg.awbc_decay_timesteps))
        if decay_steps == 0:
            return base
        progress = min(max(float(timesteps) / float(decay_steps), 0.0), 1.0)
        return base * (1.0 - progress)

    def _current_awbc_coef(self) -> float:
        return float(getattr(self, "_active_awbc_coef", self.cfg.awbc_coef))

    def _subtype_aux_loss(self, obs: Any, event_flags: Any, labels: Any, valid: Any) -> tuple[Any, float]:
        torch, nn = _torch_modules()
        logits = self.model.subtype_logits(obs, event_flags)
        if logits is None or float(self.cfg.subtype_aux_coef) <= 0.0:
            return torch.zeros((), device=obs.device), float("nan")
        valid_mask = valid.detach().float() > 0.5
        if not bool(valid_mask.any().detach().cpu().item()):
            return torch.zeros((), device=obs.device), float("nan")
        safe_labels = labels.clamp(min=0, max=int(logits.shape[1]) - 1)
        ce = nn.functional.cross_entropy(logits, safe_labels, reduction="none")
        loss = (ce * valid_mask.float()).sum() / (valid_mask.float().sum() + 1.0e-8)
        pred = torch.argmax(logits.detach(), dim=1)
        acc = float((pred[valid_mask] == safe_labels[valid_mask]).float().mean().detach().cpu().item())
        return loss, acc

    def _subtype_action_losses(
        self,
        dist: Any,
        action_masks: Any,
        labels: Any,
        valid: Any,
    ) -> tuple[Any, Any, float]:
        torch, nn = _torch_modules()
        ce_weight = max(0.0, float(self.cfg.subtype_action_ce_coef))
        margin_weight = max(0.0, float(self.cfg.subtype_action_margin_coef))
        zero = torch.zeros((), device=dist.probs.device)
        if ce_weight <= 0.0 and margin_weight <= 0.0:
            return zero, zero, float("nan")

        action_by_subtype = torch.as_tensor(
            [
                int(self.cfg.awbc_teacher_subtype_calm_action),
                int(self.cfg.awbc_teacher_subtype_particle_action),
                int(self.cfg.awbc_teacher_subtype_flux_action),
                int(self.cfg.awbc_teacher_subtype_thermal_action),
            ],
            device=dist.probs.device,
            dtype=torch.long,
        )
        n_actions = int(dist.probs.shape[1])
        safe_labels = labels.clamp(min=0, max=int(action_by_subtype.shape[0]) - 1)
        targets = action_by_subtype[safe_labels]
        base_valid = (valid.detach().float() > 0.5) & (targets >= 0) & (targets < n_actions)
        if bool(self.cfg.subtype_action_event_only):
            base_valid = base_valid & (safe_labels > 0)
        if action_masks is not None:
            safe_targets = targets.clamp(min=0, max=max(0, n_actions - 1)).reshape(-1, 1)
            target_feasible = action_masks.bool().gather(1, safe_targets).reshape(-1)
            base_valid = base_valid & target_feasible
        valid_count = float(base_valid.detach().float().sum().cpu().item())
        valid_rate = valid_count / max(float(labels.numel()), 1.0)
        if valid_count <= 0.0:
            return zero, zero, valid_rate

        supervision_mode = str(self.cfg.subtype_action_supervision_mode)
        if supervision_mode == "exact_action":
            ce_loss = nn.functional.nll_loss(
                torch.log(dist.probs.clamp_min(1.0e-8))[base_valid],
                targets[base_valid],
                reduction="mean",
            )
        elif supervision_mode == "positive_sensor_inclusion":
            safe_targets = targets.clamp(min=0, max=max(0, n_actions - 1))
            required_masks = self.candidate_masks_t[safe_targets].bool()
            candidates = self.candidate_masks_t.bool().reshape(1, n_actions, -1)
            required = required_masks.reshape(required_masks.shape[0], 1, -1)
            includes_required = ((candidates & required) == required).all(dim=2)
            if action_masks is not None:
                includes_required = includes_required & action_masks.bool()
            inclusion_mass = (dist.probs * includes_required.float()).sum(dim=1).clamp_min(1.0e-8)
            ce_loss = -torch.log(inclusion_mass[base_valid]).mean()
        else:
            raise ValueError(f"Unsupported subtype action supervision mode: {supervision_mode}")

        margin_loss = zero
        if margin_weight > 0.0:
            logits = dist.logits
            selected_logits = logits.gather(1, targets.clamp(min=0, max=max(0, n_actions - 1)).reshape(-1, 1))
            selected_logits = selected_logits.reshape(-1, 1)
            choices = action_by_subtype.reshape(1, -1).expand(labels.shape[0], -1)
            choice_valid = (choices >= 0) & (choices < n_actions) & (choices != targets.reshape(-1, 1))
            if action_masks is not None:
                safe_choices = choices.clamp(min=0, max=max(0, n_actions - 1))
                choice_valid = choice_valid & action_masks.bool().gather(1, safe_choices)
            choice_valid = choice_valid & base_valid.reshape(-1, 1)
            if bool(choice_valid.any().detach().cpu().item()):
                safe_choices = choices.clamp(min=0, max=max(0, n_actions - 1))
                other_logits = logits.gather(1, safe_choices)
                margin = max(0.0, float(self.cfg.subtype_action_margin))
                penalties = torch.relu(float(margin) + other_logits - selected_logits)
                margin_loss = (penalties * choice_valid.float()).sum() / (choice_valid.float().sum() + 1.0e-8)
        return ce_loss, margin_loss, valid_rate

    def predict_mask(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray | None = None,
        *,
        deterministic: bool = True,
        event_context: float | None = None,
        sampling_temperature: float = 1.0,
    ) -> np.ndarray:
        torch, _ = _torch_modules()
        obs_t = torch_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device)
        if action_mask is None or not bool(self.cfg.use_action_mask):
            action_mask = np.ones(self.candidate_masks_np.shape[0], dtype=bool)
        mask_t = torch_tensor(np.asarray(action_mask, dtype=bool).reshape(1, -1), device=self.device, dtype=torch.bool)
        event_value = self._event_from_obs(obs_t) if event_context is None else float(event_context)
        event_t = torch_tensor(np.asarray([event_value], dtype=np.float32), device=self.device)
        with torch.no_grad():
            action = self._subtype_router_action(obs_t, mask_t, event_t) if deterministic else None
            if action is None:
                dist = self.model.dist(obs_t, self.candidate_masks_t, mask_t, event_t)
                if deterministic or float(sampling_temperature) <= 0.0:
                    action = int(torch.argmax(dist.probs, dim=1).detach().cpu().item())
                else:
                    sample_dist = torch.distributions.Categorical(
                        logits=dist.logits / max(float(sampling_temperature), 1.0e-6)
                    )
                    action = int(sample_dist.sample().detach().cpu().item())
        return self.candidate_masks_np[action]

    def _subtype_router_action(self, obs_t: Any, mask_t: Any, event_t: Any) -> int | None:
        if not bool(self.cfg.subtype_router_enabled):
            return None
        logits = self.model.subtype_logits(obs_t, event_t)
        if logits is None:
            return None
        torch, _ = _torch_modules()
        probs = torch.softmax(logits, dim=1)
        confidence_t, subtype_t = torch.max(probs, dim=1)
        confidence = float(confidence_t.detach().cpu().item())
        if confidence < float(self.cfg.subtype_router_min_confidence):
            fallback = int(self.cfg.subtype_router_low_confidence_action)
            valid = mask_t.detach().cpu().numpy().reshape(-1).astype(bool)
            if 0 <= fallback < len(valid) and bool(valid[fallback]):
                return fallback
            return None
        subtype_id = int(subtype_t.detach().cpu().item())
        subtype_actions = {
            0: int(self.cfg.awbc_teacher_subtype_calm_action),
            1: int(self.cfg.awbc_teacher_subtype_particle_action),
            2: int(self.cfg.awbc_teacher_subtype_flux_action),
            3: int(self.cfg.awbc_teacher_subtype_thermal_action),
        }
        action = int(subtype_actions.get(subtype_id, -1))
        valid = mask_t.detach().cpu().numpy().reshape(-1).astype(bool)
        if 0 <= action < len(valid) and bool(valid[action]):
            return action
        return None

    def _event_from_obs(self, obs_t: Any) -> float:
        values = obs_t.reshape(-1)
        trailing = int(getattr(self, "context_feature_dim", 0))
        if bool(self.env_cfg.energy_account_enabled):
            trailing += 1
        if bool(self.env_cfg.include_event_flag_in_state):
            offset = trailing + 1
            if offset <= int(values.numel()):
                return float(np.clip(float(values[-offset].detach().cpu().item()), 0.0, 1.0))
        columns = tuple(str(name) for name in self.env_cfg.agent_context_columns)
        if columns:
            if "agent_context_event_alert" in columns:
                context_idx = columns.index("agent_context_event_alert")
            else:
                alert_indices = [idx for idx, name in enumerate(columns) if name.endswith("_alert")]
                if not alert_indices:
                    return 0.0
                context_idx = alert_indices[-1]
            trailing += int(bool(self.env_cfg.include_event_flag_in_state))
            offset = trailing + (len(columns) - int(context_idx))
            if offset <= int(values.numel()):
                return float(np.clip(float(values[-offset].detach().cpu().item()), 0.0, 1.0))
        return 0.0

    def save(self, path: str | Path) -> None:
        torch, _ = _torch_modules()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "cfg": asdict(self.cfg),
                "candidate_masks": self.candidate_masks_np,
                "candidate_prior_logits": self.candidate_prior_logits_np,
                "obs_dim": self.obs_dim,
                "n_sensors": len(self.sensor_specs),
                "history": self.history,
            },
            str(out),
        )

    def load_policy_checkpoint(self, path: str | Path) -> None:
        torch, _ = _torch_modules()
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        masks = np.asarray(payload.get("candidate_masks"), dtype=bool)
        if masks.shape != self.candidate_masks_np.shape or not np.array_equal(
            masks, self.candidate_masks_np
        ):
            raise ValueError("checkpoint candidate masks do not match trainer action geometry")
        if int(payload.get("obs_dim", -1)) != int(self.obs_dim):
            raise ValueError("checkpoint observation dimension does not match trainer")
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.history = list(payload.get("history", ()))

    def save_history(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def _flush_history(self) -> None:
        if not self.cfg.history_path:
            return
        self.save_history(self.cfg.history_path)

    def _make_env(self, *, seed_offset: int) -> WarmupSchedulingEnv:
        scenario_idx = self._training_scenario_cursor % len(self.training_scenarios)
        self._training_scenario_cursor += 1
        truth_df, env_cfg, oracle = self.training_scenarios[scenario_idx]
        return WarmupSchedulingEnv(
            truth_df,
            self.sensor_specs,
            self.constraints,
            replace(env_cfg, seed=int(env_cfg.seed) + int(seed_offset)),
            oracle=oracle,
        )

    def _sample_start_idx(
        self,
        steps: int,
        *,
        seed_offset: int,
        env: WarmupSchedulingEnv | None = None,
    ) -> int:
        truth_df = self.truth_df if env is None else env.truth_df
        oracle = self.oracle if env is None else env.oracle
        env_cfg = self.env_cfg if env is None else env.cfg
        global_max_start = max(0, len(truth_df) - int(steps) - int(oracle.cfg.horizon) - 1)
        min_start = max(0, int(self.cfg.train_start_min or 0))
        max_start = global_max_start
        if self.cfg.train_start_max is not None:
            max_start = min(max_start, int(self.cfg.train_start_max))
        if max_start < min_start:
            raise ValueError(f"No valid PPO training starts remain in [{min_start}, {max_start}]")
        rng = np.random.default_rng(int(self.cfg.seed) + int(seed_offset) + 81_337)
        if self.cfg.train_start_indices:
            starts = np.asarray(
                [int(x) for x in self.cfg.train_start_indices if min_start <= int(x) <= max_start],
                dtype=int,
            )
            if starts.size:
                return int(rng.choice(starts))
        event_flags = (
            truth_df[env_cfg.event_column].astype(bool).to_numpy()
            if env_cfg.event_column in truth_df.columns
            else np.zeros(len(truth_df), dtype=bool)
        )
        eligible_event_flags = np.zeros_like(event_flags, dtype=bool)
        eligible_event_flags[min_start : max_start + int(steps)] = event_flags[min_start : max_start + int(steps)]
        event_indices = np.flatnonzero(eligible_event_flags)
        if event_indices.size and rng.random() < float(self.cfg.event_start_prob):
            event_idx = int(rng.choice(event_indices))
            return int(np.clip(event_idx - int(steps) // 3, min_start, max_start))
        return int(rng.integers(min_start, max_start + 1))

    def _sampling_window_steps(self, requested_steps: int) -> int:
        episode_len = int(self.env_cfg.episode_len or requested_steps)
        return max(1, min(int(requested_steps), max(1, episode_len)))


@dataclass
class CustomPPOPolicy:
    trainer: CustomPPO
    name: str = "custom_ppo"
    deterministic: bool = True
    sampling_temperature: float = 1.0

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        action_mask = feasible_candidate_mask(env, self.trainer.candidate_masks_np)
        return self.trainer.predict_mask(
            env._state().astype(np.float32),
            action_mask,
            deterministic=bool(self.deterministic),
            event_context=env.online_event_context(),
            sampling_temperature=float(self.sampling_temperature),
        )

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


def evaluate_custom_ppo(
    *,
    trainer: CustomPPO,
    truth_df: pd.DataFrame,
    sensor_specs: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle,
    steps: int,
    start_indices: tuple[int, ...],
    policy_name: str = "custom_ppo",
    min_dwell_steps: int = 1,
    deterministic: bool = True,
    sampling_seed: int | None = None,
    sampling_temperature: float = 1.0,
) -> tuple[RolloutResult, dict[str, float | str | int]]:
    torch, _ = _torch_modules()
    torch.manual_seed(
        int(trainer.cfg.seed) + 700_000 if sampling_seed is None else int(sampling_seed)
    )
    base_policy = CustomPPOPolicy(
        trainer=trainer,
        name=str(policy_name),
        deterministic=bool(deterministic),
        sampling_temperature=float(sampling_temperature),
    )
    policy = (
        MinDwellPolicyWrapper(
            base_policy=base_policy,
            min_dwell_steps=int(min_dwell_steps),
            name=str(policy_name),
        )
        if int(min_dwell_steps) > 1
        else base_policy
    )
    rollouts = []
    for offset, start_idx in enumerate(start_indices):
        env = WarmupSchedulingEnv(
            truth_df,
            sensor_specs,
            constraints,
            replace(cfg, seed=int(cfg.seed) + int(offset)),
            oracle=oracle,
        )
        rollouts.append(
            run_policy_rollout(
                env,
                policy,
                steps=int(steps),
                start_idx=int(start_idx),
                reward_proxy_mode=str(cfg.reward_proxy_mode),
                reward_proxy_horizon=max(
                    1,
                    int(trainer.cfg.forecast_value_aux_lookahead_steps)
                    or int(trainer.cfg.greedy_lookahead_steps),
                ),
            )
        )
    result = rollouts[0] if len(rollouts) == 1 else concat_rollout_results(rollouts, policy_name=policy.name)
    return result, rollout_metrics(result)


def feasible_candidate_mask(env: WarmupSchedulingEnv, candidate_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
    feasible = np.zeros(masks.shape[0], dtype=bool)
    for idx, mask in enumerate(masks):
        feasible[idx] = bool(env.is_mask_executable(mask))
    if not np.any(feasible):
        # Preserve the historical fallback only when the projector has no
        # feasible candidate.  Under an active dwell hold, the previous
        # executed subset must remain represented by the candidate geometry.
        if int(getattr(env, "dwell_hold_remaining", 0)) <= 0:
            feasible[:] = True
        else:
            raise ValueError(
                "candidate_masks must contain the currently executed subset "
                "while minimum dwell is active"
            )
    return feasible


def oracle_greedy_candidate_index(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    lookahead_steps: int,
) -> int:
    costs = oracle_greedy_candidate_costs(env, candidate_masks, lookahead_steps=lookahead_steps)
    return int(np.argmin(costs))


def oracle_greedy_candidate_costs(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    lookahead_steps: int,
) -> np.ndarray:
    snapshot = snapshot_env(env)
    feasible = feasible_candidate_mask(env, candidate_masks)
    costs_by_action = np.full(len(candidate_masks), np.inf, dtype=np.float32)
    for idx, mask in enumerate(np.asarray(candidate_masks, dtype=bool)):
        if not feasible[idx]:
            continue
        restore_env(env, snapshot)
        costs: list[float] = []
        for _ in range(max(1, int(lookahead_steps))):
            _, _, done, info = env.step_mask(mask)
            oracle_loss = float(info.get("oracle_loss", float("inf")))
            shaping = float(info.get("shaping_penalty", 0.0))
            cost = oracle_loss + shaping
            if np.isfinite(cost):
                costs.append(float(cost))
            if done:
                break
        avg_cost = float(np.mean(costs)) if costs else float("inf")
        costs_by_action[idx] = float(avg_cost)
    restore_env(env, snapshot)
    return costs_by_action


def forecast_block_gain(
    env: WarmupSchedulingEnv,
    selected_mask: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    horizon: int,
    relative: bool = False,
) -> float:
    """Return baseline block loss minus selected block loss from one state.

    Both branches use the same snapshot and random-number state.  The baseline
    is the previously executed subset, so the target measures the incremental
    value of changing the action over the full minimum-dwell block.
    """
    snapshot = snapshot_env(env)

    def block_cost(mask: np.ndarray) -> float:
        restore_env(env, snapshot)
        costs: list[float] = []
        for _ in range(max(1, int(horizon))):
            _, _, done, info = env.step_mask(np.asarray(mask, dtype=bool))
            value = float(info.get("oracle_loss", float("nan")))
            if np.isfinite(value):
                costs.append(value)
            if done:
                break
        return float(np.mean(costs)) if costs else float("nan")

    selected_cost = block_cost(selected_mask)
    baseline_cost = block_cost(baseline_mask)
    restore_env(env, snapshot)
    if not np.isfinite(selected_cost) or not np.isfinite(baseline_cost):
        return 0.0
    gain = float(baseline_cost - selected_cost)
    if bool(relative):
        scale = max(abs(float(baseline_cost)) + abs(float(selected_cost)), 1.0e-6)
        return float(np.clip(2.0 * gain / scale, -1.0, 1.0))
    return gain


def forecast_block_cost(
    env: WarmupSchedulingEnv,
    selected_mask: np.ndarray,
    *,
    horizon: int,
) -> float:
    """Return the mean frozen-forecaster loss over one executable dwell block."""
    snapshot = snapshot_env(env)
    costs: list[float] = []
    try:
        for _ in range(max(1, int(horizon))):
            _, _, done, info = env.step_mask(np.asarray(selected_mask, dtype=bool))
            value = float(info.get("oracle_loss", float("nan")))
            if np.isfinite(value):
                costs.append(value)
            if done:
                break
    finally:
        restore_env(env, snapshot)
    return float(np.mean(costs)) if costs else float("inf")


def soft_forecast_value_targets(
    costs: np.ndarray,
    feasible: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    values = np.asarray(costs, dtype=np.float64)
    valid = np.asarray(feasible, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("costs and feasible must be matched two-dimensional arrays")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    targets = np.zeros_like(values, dtype=np.float64)
    for row in range(values.shape[0]):
        row_valid = valid[row]
        if not np.any(row_valid):
            raise ValueError("every row requires at least one finite feasible action")
        row_costs = values[row, row_valid]
        scale = max(float(np.std(row_costs)), 1.0e-8)
        logits = -(row_costs - float(np.mean(row_costs))) / (scale * float(temperature))
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        targets[row, row_valid] = weights / float(np.sum(weights))
    return targets.astype(np.float32)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for step in reversed(range(rewards.shape[0])):
        next_value = float(last_value) if step == rewards.shape[0] - 1 else float(values[step + 1])
        next_nonterminal = 1.0 - float(dones[step])
        delta = float(rewards[step]) + float(gamma) * next_value * next_nonterminal - float(values[step])
        last_gae = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_gae
        advantages[step] = float(last_gae)
    return advantages


def build_future_soc_targets(
    soc_ratios: np.ndarray,
    episode_ids: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.asarray(soc_ratios).shape[0])
    h = max(0, int(horizon))
    targets = np.zeros((n, h), dtype=np.float32)
    mask = np.zeros((n, h), dtype=np.float32)
    if n == 0 or h <= 0:
        return targets, mask
    soc = np.asarray(soc_ratios, dtype=np.float32).reshape(-1)
    episodes = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    for i in range(n):
        end = min(n, i + h)
        width = end - i
        if width <= 0:
            continue
        same_episode = episodes[i:end] == episodes[i]
        if not np.any(same_episode):
            continue
        targets[i, :width] = soc[i:end]
        mask[i, :width] = same_episode.astype(np.float32)
    return targets, mask


def advantage_weighted_bc_loss(dist: Any, greedy_actions: Any, advantages: Any, label_mask: Any | None = None) -> Any:
    positive = advantages.detach().clamp_min(0.0)
    if label_mask is not None:
        positive = positive * label_mask.detach().float()
    log_prob = dist.log_prob(greedy_actions)
    denom = positive.sum().clamp_min(1.0e-8)
    return -(positive * log_prob).sum() / denom


def prior_kl_regularizer(dist: Any, prior_logits: Any | None, action_mask: Any) -> Any:
    torch, _ = _torch_modules()
    if prior_logits is None:
        return torch.zeros((), device=dist.probs.device)
    logits = prior_logits.reshape(1, -1).expand_as(dist.probs)
    valid = action_mask.bool()
    if valid.ndim == 1:
        valid = valid.reshape(1, -1).expand_as(dist.probs)
    logits = logits.masked_fill(~valid, -1.0e9)
    prior_probs = torch.softmax(logits, dim=1)
    prior_log = torch.log(prior_probs.clamp_min(1.0e-8))
    current_log = torch.log(dist.probs.clamp_min(1.0e-8))
    kl = (prior_probs * (prior_log - current_log)).sum(dim=1)
    return kl.mean()


def snapshot_env(env: WarmupSchedulingEnv) -> dict[str, object]:
    return {
        "episode_start_idx": int(env.episode_start_idx),
        "episode_end_idx": int(env.episode_end_idx),
        "current_idx": int(env.current_idx),
        "last_observation": np.array(env.last_observation, copy=True),
        "observed_mask": np.array(env.observed_mask, copy=True),
        "history": np.array(env.history, copy=True),
        "mask_history": np.array(env.mask_history, copy=True),
        "posterior_variance": np.array(env.posterior_variance, copy=True),
        "previous_action_mask": np.array(env.previous_action_mask, copy=True),
        "sensor_on_counts": np.array(env.sensor_on_counts, copy=True),
        "elapsed_steps": int(env.elapsed_steps),
        "dwell_hold_remaining": int(env.dwell_hold_remaining),
        "current_energy": float(env.current_energy),
        "energy_deficit_steps": int(env.energy_deficit_steps),
        "energy_deficit_total": float(env.energy_deficit_total),
        "last_info": copy.deepcopy(env.last_info),
        "rng_state": copy.deepcopy(env.rng.bit_generator.state),
        "runtimes": {
            sensor_id: {
                "mode": runtime.mode,
                "warm_remaining": int(runtime.warm_remaining),
                "last_observed_step": runtime.last_observed_step,
                "warmup_abort_count": int(runtime.warmup_abort_count),
            }
            for sensor_id, runtime in env.runtimes.items()
        },
    }


def restore_env(env: WarmupSchedulingEnv, snapshot: dict[str, object]) -> None:
    env.episode_start_idx = int(snapshot["episode_start_idx"])
    env.episode_end_idx = int(snapshot["episode_end_idx"])
    env.current_idx = int(snapshot["current_idx"])
    env.last_observation = np.asarray(snapshot["last_observation"], dtype=float).copy()
    env.observed_mask = np.asarray(snapshot["observed_mask"], dtype=float).copy()
    env.history = np.asarray(snapshot["history"], dtype=float).copy()
    env.mask_history = np.asarray(snapshot["mask_history"], dtype=float).copy()
    env.posterior_variance = np.asarray(snapshot["posterior_variance"], dtype=float).copy()
    env.previous_action_mask = np.asarray(snapshot["previous_action_mask"], dtype=float).copy()
    env.sensor_on_counts = np.asarray(snapshot.get("sensor_on_counts", np.zeros(len(env.sensor_specs))), dtype=float).copy()
    env.elapsed_steps = int(snapshot.get("elapsed_steps", 0))
    env.dwell_hold_remaining = int(snapshot.get("dwell_hold_remaining", 0))
    env.current_energy = float(snapshot.get("current_energy", 0.0))
    env.energy_deficit_steps = int(snapshot.get("energy_deficit_steps", 0))
    env.energy_deficit_total = float(snapshot.get("energy_deficit_total", 0.0))
    env.last_info = copy.deepcopy(snapshot["last_info"])
    env.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
    runtime_state = snapshot["runtimes"]
    assert isinstance(runtime_state, dict)
    for sensor_id, state in runtime_state.items():
        assert isinstance(state, dict)
        runtime = env.runtimes[str(sensor_id)]
        runtime.mode = state["mode"]
        runtime.warm_remaining = int(state["warm_remaining"])
        runtime.last_observed_step = state["last_observed_step"]
        runtime.warmup_abort_count = int(state["warmup_abort_count"])


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


def torch_tensor(array: Any, *, device: Any, dtype: Any | None = None) -> Any:
    torch, _ = _torch_modules()
    return torch.as_tensor(array, dtype=dtype if dtype is not None else torch.float32, device=device)


def torch_concat(tensors: list[Any], *, dim: int) -> Any:
    torch, _ = _torch_modules()
    return torch.cat(tensors, dim=int(dim))


def _mean_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(row[key]) for row in rows]))
