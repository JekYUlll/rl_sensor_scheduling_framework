from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv
from v2.oracle import LinearFrozenForecastOracle
from v2.policies import MinDwellPolicyWrapper
from v2.power_projector import PowerConstraintsV2
from v2.rollout import RolloutResult, concat_rollout_results, rollout_metrics, run_policy_rollout
from v2.sensor_spec import SensorSpecV2


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
    vf_coef: float = 0.5
    awbc_coef: float = 0.1
    awbc_decay_timesteps: int = 0
    awbc_label_stride: int = 4
    bc_pretrain_steps: int = 0
    bc_pretrain_epochs: int = 4
    bc_pretrain_batch_size: int = 128
    bc_pretrain_loss_coef: float = 1.0
    subtype_aux_coef: float = 0.0
    subtype_aux_classes: int = 4
    subtype_aux_lookahead_steps: int = 0
    subtype_action_ce_coef: float = 0.0
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
    trainable_action_prior: bool = True
    event_aware_critic: bool = True
    event_gated_actor: bool = False
    context_encoder_enabled: bool = False
    context_feature_dim: int = 0
    context_hidden_dim: int = 64
    context_fusion_mode: str = "concat"
    context_layer_norm: bool = False
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
    def __new__(cls, n_sensors: int, embed_dim: int) -> Any:
        torch, nn = _torch_modules()

        class _ActionEmbedding(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sensor_embedding = SensorEmbedding(int(n_sensors), int(embed_dim))

            def forward(self, action_masks: Any) -> Any:
                masks = action_masks.float()
                sensor_ids = torch.arange(int(n_sensors), device=masks.device, dtype=torch.long)
                sensor_emb = self.sensor_embedding(sensor_ids)
                return masks @ sensor_emb

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
        trainable_action_prior: bool = True,
        event_gated: bool = False,
        subtype_aux_classes: int = 0,
        context_encoder_enabled: bool = False,
        context_feature_dim: int = 0,
        context_hidden_dim: int = 64,
        context_fusion_mode: str = "concat",
        context_layer_norm: bool = False,
    ) -> Any:
        torch, nn = _torch_modules()

        class _MaskedActor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_action_embedding = bool(use_action_embedding)
                self.event_gated = bool(event_gated)
                self.n_actions = int(n_actions or 0)
                self.subtype_aux_classes = max(0, int(subtype_aux_classes))
                self.context_fusion_mode = str(context_fusion_mode)
                self.context_feature_dim = (
                    max(0, min(int(context_feature_dim), int(obs_dim) - 1))
                    if bool(context_encoder_enabled)
                    else 0
                )
                main_obs_dim = int(obs_dim) - int(self.context_feature_dim)
                if self.use_action_embedding:
                    self.action_embedding = ActionEmbedding(int(n_sensors), int(embed_dim))
                else:
                    if self.n_actions <= 0:
                        raise ValueError("n_actions must be provided when use_action_embedding=False")
                    self.action_embedding = nn.Embedding(self.n_actions, int(embed_dim))
                    nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.08)
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
                    if self.subtype_aux_classes > 0
                    else None
                )
                self.action_bias = nn.Linear(int(embed_dim), 1)
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

            def encode_context(
                self,
                obs: Any,
                event_flag: Any | None = None,
            ) -> Any:
                main_obs, context_obs = self._split_obs(obs)
                context = self.obs_encoder(main_obs)
                if self.context_encoder is not None and context_obs is not None:
                    context_extra = self.context_encoder(context_obs)
                    if self.context_fusion_mode == "gated_add" and self.context_gate is not None:
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
                action_emb = self._action_embeddings(candidate_masks)
                logits = context @ action_emb.transpose(0, 1) / max(float(action_emb.shape[-1]) ** 0.5, 1.0)
                logits = logits + self.action_bias(action_emb).reshape(1, -1)
                if self.action_prior is not None:
                    logits = logits + self.action_prior.reshape(1, -1)
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
                    trainable_action_prior=bool(trainable_action_prior),
                    event_gated=bool(event_gated_actor),
                    subtype_aux_classes=int(subtype_aux_classes),
                    context_encoder_enabled=bool(context_encoder_enabled),
                    context_feature_dim=int(context_feature_dim),
                    context_hidden_dim=int(context_hidden_dim),
                    context_fusion_mode=str(context_fusion_mode),
                    context_layer_norm=bool(context_layer_norm),
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
    ) -> None:
        torch, _ = _torch_modules()
        self.truth_df = truth_df
        self.sensor_specs = list(sensor_specs)
        self.constraints = constraints
        self.env_cfg = env_cfg
        self.oracle = oracle
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
        self.obs_dim = int(np.asarray(probe_obs).shape[0])
        self.context_feature_dim = int(cfg.context_feature_dim)
        if self.context_feature_dim <= 0:
            self.context_feature_dim = int(getattr(probe_env, "alert_context_feature_dim", 0))
        if bool(cfg.context_encoder_enabled) and self.context_feature_dim <= 0:
            raise ValueError("context_encoder_enabled requires alert/context features in the environment state")
        if self.context_feature_dim > 0:
            self.cfg = replace(self.cfg, context_feature_dim=int(self.context_feature_dim))
        self.model = ActorCritic(
            obs_dim=self.obs_dim,
            n_sensors=len(self.sensor_specs),
            embed_dim=int(cfg.embed_dim),
            hidden_dim=int(cfg.hidden_dim),
            n_actions=int(self.candidate_masks_np.shape[0]),
            candidate_prior_logits=self.candidate_prior_logits_np,
            use_action_embedding=bool(cfg.use_action_embedding),
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
        total_timesteps = int(self.cfg.total_timesteps)
        steps_done = 0
        update_idx = 0
        while steps_done < total_timesteps:
            self._active_awbc_coef = self._effective_awbc_coef(steps_done)
            rollout_steps = min(int(self.cfg.n_steps), total_timesteps - steps_done)
            batch = self.collect_rollout(int(rollout_steps), seed_offset=update_idx * 997)
            metrics = self.update(batch)
            steps_done += int(batch["obs"].shape[0])
            update_idx += 1
            metrics["timesteps"] = int(steps_done)
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
                f"awbc_coef={float(metrics['awbc_coef']):.6f} "
                f"awbc_label_rate={float(metrics['awbc_label_rate']):.3f}",
                flush=True,
            )
        return self

    def collect_rollout(self, n_steps: int, *, seed_offset: int = 0) -> dict[str, np.ndarray]:
        torch, _ = _torch_modules()
        env = self._make_env(seed_offset=seed_offset)
        env.reset(start_idx=self._sample_start_idx(self._sampling_window_steps(int(n_steps)), seed_offset=seed_offset))

        obs_rows: list[np.ndarray] = []
        action_rows: list[int] = []
        greedy_rows: list[int] = []
        logprob_rows: list[float] = []
        reward_rows: list[float] = []
        done_rows: list[float] = []
        value_rows: list[float] = []
        event_rows: list[float] = []
        action_mask_rows: list[np.ndarray] = []
        awbc_valid_rows: list[float] = []
        subtype_label_rows: list[int] = []
        subtype_valid_rows: list[float] = []
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
            should_label = float(self._current_awbc_coef()) > 0.0 and (step % label_stride == 0)
            if should_label:
                greedy = self._awbc_teacher_action(env, action_mask_np)
                awbc_valid = 1.0
            else:
                greedy = action
                awbc_valid = 0.0
            _, reward, done, info = env.step_mask(self.candidate_masks_np[action])

            obs_rows.append(obs_np)
            action_rows.append(action)
            greedy_rows.append(int(greedy))
            logprob_rows.append(float(logprob_t.detach().cpu().item()))
            reward_rows.append(float(reward))
            done_rows.append(float(done))
            value_rows.append(float(value_t.detach().cpu().item()))
            event_rows.append(event_flag)
            action_mask_rows.append(action_mask_np.astype(bool))
            awbc_valid_rows.append(float(awbc_valid))
            subtype_label_rows.append(int(subtype_label))
            subtype_valid_rows.append(float(subtype_valid))
            soc_rows.append(float(info.get("soc_ratio", 1.0)))
            episode_rows.append(int(episode_id))
            last_done = bool(done)
            if done and step < int(n_steps) - 1:
                env = self._make_env(seed_offset=seed_offset + step + 1)
                env.reset(
                    start_idx=self._sample_start_idx(
                        self._sampling_window_steps(int(n_steps)),
                        seed_offset=seed_offset + step + 1,
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
            "event_flags": np.asarray(event_rows, dtype=np.float32),
            "action_masks": np.vstack(action_mask_rows).astype(bool),
            "awbc_valid": np.asarray(awbc_valid_rows, dtype=np.float32),
            "subtype_labels": np.asarray(subtype_label_rows, dtype=np.int64),
            "subtype_valid": np.asarray(subtype_valid_rows, dtype=np.float32),
            "soc_aux_targets": soc_targets.astype(np.float32),
            "soc_aux_mask": soc_mask.astype(np.float32),
        }

    def collect_teacher_batch(self, n_steps: int, *, seed_offset: int = 0) -> dict[str, np.ndarray]:
        env = self._make_env(seed_offset=seed_offset)
        env.reset(start_idx=self._sample_start_idx(self._sampling_window_steps(int(n_steps)), seed_offset=seed_offset))

        obs_rows: list[np.ndarray] = []
        teacher_rows: list[int] = []
        event_rows: list[float] = []
        action_mask_rows: list[np.ndarray] = []
        subtype_label_rows: list[int] = []
        subtype_valid_rows: list[float] = []
        episode_id = 0
        for step in range(int(n_steps)):
            obs_np = env._state().astype(np.float32)
            action_mask_np = (
                feasible_candidate_mask(env, self.candidate_masks_np)
                if bool(self.cfg.use_action_mask)
                else np.ones(self.candidate_masks_np.shape[0], dtype=bool)
            )
            teacher = int(self._awbc_teacher_action(env, action_mask_np))
            if not (0 <= teacher < len(action_mask_np) and bool(action_mask_np[teacher])):
                feasible = np.flatnonzero(action_mask_np)
                teacher = int(feasible[0]) if feasible.size else 0
            subtype_label, subtype_valid = self._subtype_aux_label(env)

            obs_rows.append(obs_np)
            teacher_rows.append(int(teacher))
            event_rows.append(float(env.online_event_context()))
            action_mask_rows.append(action_mask_np.astype(bool))
            subtype_label_rows.append(int(subtype_label))
            subtype_valid_rows.append(float(subtype_valid))

            _, _, done, _ = env.step_mask(self.candidate_masks_np[teacher])
            if done and step < int(n_steps) - 1:
                env = self._make_env(seed_offset=seed_offset + step + 1)
                env.reset(
                    start_idx=self._sample_start_idx(
                        self._sampling_window_steps(int(n_steps)),
                        seed_offset=seed_offset + step + 1,
                    )
                )
                episode_id += 1

        return {
            "obs": np.vstack(obs_rows).astype(np.float32),
            "teacher_actions": np.asarray(teacher_rows, dtype=np.int64),
            "event_flags": np.asarray(event_rows, dtype=np.float32),
            "action_masks": np.vstack(action_mask_rows).astype(bool),
            "subtype_labels": np.asarray(subtype_label_rows, dtype=np.int64),
            "subtype_valid": np.asarray(subtype_valid_rows, dtype=np.float32),
            "episode_ids": np.full(len(teacher_rows), int(episode_id), dtype=np.int64),
        }

    def bc_pretrain(self, n_steps: int) -> dict[str, float | int]:
        torch, nn = _torch_modules()
        batch = self.collect_teacher_batch(int(n_steps), seed_offset=91_000)
        obs = torch_tensor(batch["obs"], device=self.device)
        teacher_actions = torch_tensor(batch["teacher_actions"], device=self.device, dtype=torch.long)
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
                mb_events = event_flags[idx]
                mb_masks = action_masks[idx]
                mb_subtype_labels = subtype_labels[idx]
                mb_subtype_valid = subtype_valid[idx]
                dist = self.model.dist(mb_obs, self.candidate_masks_t, mb_masks, mb_events)
                bc_loss = -dist.log_prob(mb_teacher).mean()
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
        event_flags = torch_tensor(batch["event_flags"], device=self.device)
        action_masks = torch_tensor(batch["action_masks"], device=self.device, dtype=torch.bool)
        awbc_valid = torch_tensor(batch["awbc_valid"], device=self.device)
        subtype_labels = torch_tensor(batch["subtype_labels"], device=self.device, dtype=torch.long)
        subtype_valid = torch_tensor(batch["subtype_valid"], device=self.device)
        soc_aux_targets = torch_tensor(batch["soc_aux_targets"], device=self.device)
        soc_aux_mask = torch_tensor(batch["soc_aux_mask"], device=self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

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
                mb_advantages = advantages[idx]
                mb_events = event_flags[idx]
                mb_masks = action_masks[idx]
                mb_awbc_valid = awbc_valid[idx]
                mb_subtype_labels = subtype_labels[idx]
                mb_subtype_valid = subtype_valid[idx]
                mb_soc_targets = soc_aux_targets[idx]
                mb_soc_mask = soc_aux_mask[idx]

                dist = self.model.dist(mb_obs, self.candidate_masks_t, mb_masks, mb_events)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                values = self.model.value(mb_obs, mb_events)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - float(self.cfg.clip_range),
                    1.0 + float(self.cfg.clip_range),
                )
                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
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
                loss = (
                    policy_loss
                    + float(self.cfg.vf_coef) * value_loss
                    - float(self.cfg.ent_coef) * entropy
                    + float(self._current_awbc_coef()) * awbc_loss
                    + float(self.cfg.prior_kl_coef) * prior_kl_loss
                    + float(self.cfg.soc_aux_coef) * soc_aux_loss
                    + float(self.cfg.subtype_aux_coef) * subtype_aux_loss
                    + float(self.cfg.subtype_action_ce_coef) * subtype_action_ce_loss
                    + float(self.cfg.subtype_action_margin_coef) * subtype_action_margin_loss
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
                        "entropy": float(entropy.detach().cpu().item()),
                        "awbc_loss": float(awbc_loss.detach().cpu().item()),
                        "prior_kl_loss": float(prior_kl_loss.detach().cpu().item()),
                        "soc_aux_loss": float(soc_aux_loss.detach().cpu().item()),
                        "subtype_aux_loss": float(subtype_aux_loss.detach().cpu().item()),
                        "subtype_aux_accuracy": float(subtype_aux_acc),
                        "subtype_action_ce_loss": float(subtype_action_ce_loss.detach().cpu().item()),
                        "subtype_action_margin_loss": float(subtype_action_margin_loss.detach().cpu().item()),
                        "subtype_action_valid_rate": float(subtype_action_valid_rate),
                    }
                )

        return {
            "loss": _mean_metric(loss_rows, "loss"),
            "policy_loss": _mean_metric(loss_rows, "policy_loss"),
            "value_loss": _mean_metric(loss_rows, "value_loss"),
            "entropy": _mean_metric(loss_rows, "entropy"),
            "awbc_loss": _mean_metric(loss_rows, "awbc_loss"),
            "prior_kl_loss": _mean_metric(loss_rows, "prior_kl_loss"),
            "soc_aux_loss": _mean_metric(loss_rows, "soc_aux_loss"),
            "subtype_aux_loss": _mean_metric(loss_rows, "subtype_aux_loss"),
            "subtype_aux_accuracy": _mean_metric(loss_rows, "subtype_aux_accuracy"),
            "subtype_action_ce_loss": _mean_metric(loss_rows, "subtype_action_ce_loss"),
            "subtype_action_margin_loss": _mean_metric(loss_rows, "subtype_action_margin_loss"),
            "subtype_action_valid_rate": _mean_metric(loss_rows, "subtype_action_valid_rate"),
            "advantage_mean": float(np.mean(batch["advantages"])),
            "advantage_std": float(np.std(batch["advantages"])),
            "event_rate": float(np.mean(batch["event_flags"])),
            "awbc_label_rate": float(np.mean(batch["awbc_valid"])),
            "awbc_coef": float(self._current_awbc_coef()),
            "greedy_unique_actions": int(np.unique(batch["greedy_actions"]).size),
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
        if action_masks is not None:
            safe_targets = targets.clamp(min=0, max=max(0, n_actions - 1)).reshape(-1, 1)
            target_feasible = action_masks.bool().gather(1, safe_targets).reshape(-1)
            base_valid = base_valid & target_feasible
        valid_count = float(base_valid.detach().float().sum().cpu().item())
        valid_rate = valid_count / max(float(labels.numel()), 1.0)
        if valid_count <= 0.0:
            return zero, zero, valid_rate

        ce_loss = nn.functional.nll_loss(
            torch.log(dist.probs.clamp_min(1.0e-8))[base_valid],
            targets[base_valid],
            reduction="mean",
        )

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
                action = int(torch.argmax(dist.probs, dim=1).detach().cpu().item()) if deterministic else int(dist.sample().detach().cpu().item())
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

    def save_history(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def _flush_history(self) -> None:
        if not self.cfg.history_path:
            return
        self.save_history(self.cfg.history_path)

    def _make_env(self, *, seed_offset: int) -> WarmupSchedulingEnv:
        return WarmupSchedulingEnv(
            self.truth_df,
            self.sensor_specs,
            self.constraints,
            replace(self.env_cfg, seed=int(self.env_cfg.seed) + int(seed_offset)),
            oracle=self.oracle,
        )

    def _sample_start_idx(self, steps: int, *, seed_offset: int) -> int:
        global_max_start = max(0, len(self.truth_df) - int(steps) - int(self.oracle.cfg.horizon) - 1)
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
            self.truth_df[self.env_cfg.event_column].astype(bool).to_numpy()
            if self.env_cfg.event_column in self.truth_df.columns
            else np.zeros(len(self.truth_df), dtype=bool)
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

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        action_mask = feasible_candidate_mask(env, self.trainer.candidate_masks_np)
        return self.trainer.predict_mask(
            env._state().astype(np.float32),
            action_mask,
            deterministic=bool(self.deterministic),
            event_context=env.online_event_context(),
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
) -> tuple[RolloutResult, dict[str, float | str | int]]:
    base_policy = CustomPPOPolicy(trainer=trainer, name=str(policy_name))
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
        rollouts.append(run_policy_rollout(env, policy, steps=int(steps), start_idx=int(start_idx)))
    result = rollouts[0] if len(rollouts) == 1 else concat_rollout_results(rollouts, policy_name=policy.name)
    return result, rollout_metrics(result)


def feasible_candidate_mask(env: WarmupSchedulingEnv, candidate_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
    feasible = np.zeros(masks.shape[0], dtype=bool)
    for idx, mask in enumerate(masks):
        result = env.projector.project_mask(mask, env.runtimes)
        feasible[idx] = bool(result.feasible and np.array_equal(result.selected_mask.astype(bool), mask))
    if not np.any(feasible):
        feasible[:] = True
    return feasible


def oracle_greedy_candidate_index(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    lookahead_steps: int,
) -> int:
    snapshot = snapshot_env(env)
    feasible = feasible_candidate_mask(env, candidate_masks)
    best_idx = int(np.flatnonzero(feasible)[0])
    best_cost = float("inf")
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
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_idx = int(idx)
    restore_env(env, snapshot)
    return int(best_idx)


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
