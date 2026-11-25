from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv
from v2.oracle import LinearFrozenForecastOracle
from v2.power_projector import PowerConstraintsV2
from v2.rollout import RolloutResult, rollout_metrics
from v2.sensor_spec import SensorSpecV2
from v2.warmup_state import SensorMode

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised only in minimal local envs.
    gym = None

_GymBase = gym.Env if gym is not None else object


class V2GymEnv(_GymBase):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        truth_df: pd.DataFrame,
        sensor_specs: list[SensorSpecV2],
        constraints: PowerConstraintsV2,
        cfg: WarmupEnvConfig,
        oracle: LinearFrozenForecastOracle,
        random_start: bool = True,
        seed: int = 42,
        active_score_bonus: float = 0.08,
        warming_score_bonus: float = 0.16,
        event_start_prob: float = 0.35,
        sensor_score_prior: np.ndarray | None = None,
        event_sensor_score_prior: np.ndarray | None = None,
        non_event_sensor_score_prior: np.ndarray | None = None,
        action_scale: float = 0.35,
        candidate_masks: np.ndarray | None = None,
    ) -> None:
        from gymnasium import spaces

        if gym is not None:
            super().__init__()
        self.truth_df = truth_df
        self.sensor_specs = list(sensor_specs)
        self.constraints = constraints
        self.cfg = cfg
        self.oracle = oracle
        self.random_start = bool(random_start)
        self.active_score_bonus = float(active_score_bonus)
        self.warming_score_bonus = float(warming_score_bonus)
        self.event_start_prob = float(event_start_prob)
        self.action_scale = float(action_scale)
        self.candidate_masks = (
            None
            if candidate_masks is None
            else np.asarray(candidate_masks, dtype=bool).reshape(-1, len(self.sensor_specs))
        )
        self.sensor_score_prior = (
            np.zeros(len(self.sensor_specs), dtype=float)
            if sensor_score_prior is None
            else np.asarray(sensor_score_prior, dtype=float).reshape(len(self.sensor_specs))
        )
        self.event_sensor_score_prior = (
            np.zeros(len(self.sensor_specs), dtype=float)
            if event_sensor_score_prior is None
            else np.asarray(event_sensor_score_prior, dtype=float).reshape(len(self.sensor_specs))
        )
        self.non_event_sensor_score_prior = (
            np.zeros(len(self.sensor_specs), dtype=float)
            if non_event_sensor_score_prior is None
            else np.asarray(non_event_sensor_score_prior, dtype=float).reshape(len(self.sensor_specs))
        )
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)
        self.env = WarmupSchedulingEnv(
            truth_df,
            self.sensor_specs,
            constraints,
            cfg,
            oracle=oracle,
        )
        obs, _ = self.env.reset()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )
        if self.candidate_masks is None:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self.sensor_specs),),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(int(self.candidate_masks.shape[0]))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)
        options = options or {}
        start_idx = options.get("start_idx")
        if start_idx is None:
            max_start = max(0, len(self.truth_df) - int(self.cfg.episode_len or len(self.truth_df)) - int(self.oracle.cfg.horizon) - 1)
            start_idx = self._sample_start_idx(max_start) if self.random_start and max_start > 0 else 0
        obs, info = self.env.reset(start_idx=int(start_idx))
        return obs.astype(np.float32), info

    def step(self, action):
        if self.candidate_masks is None:
            obs, reward, done, info = self.env.step_scores(self._apply_retention_bonus(np.asarray(action, dtype=float)))
        else:
            action_idx = int(np.asarray(action).reshape(-1)[0])
            action_idx = int(np.clip(action_idx, 0, int(self.candidate_masks.shape[0]) - 1))
            obs, reward, done, info = self.env.step_mask(self.candidate_masks[action_idx])
        return obs.astype(np.float32), float(reward), bool(done), False, info

    def _sample_start_idx(self, max_start: int) -> int:
        event_indices = np.flatnonzero(getattr(self.env, "event_flags", np.asarray([], dtype=bool)))
        if event_indices.size and self.rng.random() < self.event_start_prob:
            event_idx = int(self.rng.choice(event_indices))
            offset = int(self.rng.integers(0, max(1, int(self.cfg.episode_len or len(self.truth_df)))))
            return int(np.clip(event_idx - offset, 0, max_start))
        return int(self.rng.integers(0, max_start + 1))

    def _apply_retention_bonus(self, action: np.ndarray) -> np.ndarray:
        scores = float(self.action_scale) * np.asarray(action, dtype=float).reshape(-1).copy()
        scores += self.sensor_score_prior
        scores += self._context_prior()
        for idx, sensor_id in enumerate(self.env.sensor_ids):
            mode = self.env.runtimes[sensor_id].mode
            if mode == SensorMode.WARMING:
                scores[idx] += self.warming_score_bonus
            elif mode == SensorMode.ACTIVE:
                scores[idx] += self.active_score_bonus
        return scores

    def _context_prior(self) -> np.ndarray:
        flags = getattr(self.env, "event_flags", np.asarray([], dtype=bool))
        idx = int(getattr(self.env, "current_idx", 0))
        is_event = bool(idx < len(flags) and flags[idx])
        return self.event_sensor_score_prior if is_event else self.non_event_sensor_score_prior


@dataclass
class SB3ProjectedPPOPolicy:
    model: object
    name: str = "ppo"
    deterministic: bool = True
    active_score_bonus: float = 0.08
    warming_score_bonus: float = 0.16
    sensor_score_prior: np.ndarray | None = None
    event_sensor_score_prior: np.ndarray | None = None
    non_event_sensor_score_prior: np.ndarray | None = None
    action_scale: float = 0.35
    candidate_masks: np.ndarray | None = None

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray | None:
        if self.candidate_masks is None:
            return None
        obs = env._state().astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=bool(self.deterministic))
        masks = np.asarray(self.candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
        action_idx = int(np.asarray(action).reshape(-1)[0])
        action_idx = int(np.clip(action_idx, 0, int(masks.shape[0]) - 1))
        return masks[action_idx]

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        obs = env._state().astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=bool(self.deterministic))
        if self.candidate_masks is not None:
            masks = np.asarray(self.candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
            action_idx = int(np.asarray(action).reshape(-1)[0])
            action_idx = int(np.clip(action_idx, 0, int(masks.shape[0]) - 1))
            return np.where(masks[action_idx], 1.0, -1.0)
        scores = float(self.action_scale) * np.asarray(action, dtype=float).reshape(-1).copy()
        if self.sensor_score_prior is not None:
            scores += np.asarray(self.sensor_score_prior, dtype=float).reshape(-1)
        if _is_event_context(env):
            if self.event_sensor_score_prior is not None:
                scores += np.asarray(self.event_sensor_score_prior, dtype=float).reshape(-1)
        elif self.non_event_sensor_score_prior is not None:
            scores += np.asarray(self.non_event_sensor_score_prior, dtype=float).reshape(-1)
        for idx, sensor_id in enumerate(env.sensor_ids):
            mode = env.runtimes[sensor_id].mode
            if mode == SensorMode.WARMING:
                scores[idx] += self.warming_score_bonus
            elif mode == SensorMode.ACTIVE:
                scores[idx] += self.active_score_bonus
        return scores


def _is_event_context(env: WarmupSchedulingEnv) -> bool:
    flags = getattr(env, "event_flags", np.asarray([], dtype=bool))
    idx = int(getattr(env, "current_idx", 0))
    return bool(idx < len(flags) and flags[idx])


def make_vec_env(
    *,
    truth_df: pd.DataFrame,
    sensor_specs: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle,
    n_envs: int,
    seed: int,
    vec_type: str = "subproc",
    active_score_bonus: float = 0.08,
    warming_score_bonus: float = 0.16,
    event_start_prob: float = 0.35,
    sensor_score_prior: np.ndarray | None = None,
    event_sensor_score_prior: np.ndarray | None = None,
    non_event_sensor_score_prior: np.ndarray | None = None,
    action_scale: float = 0.35,
    candidate_masks: np.ndarray | None = None,
):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def make_one(rank: int) -> Callable[[], V2GymEnv]:
        def _factory() -> V2GymEnv:
            return V2GymEnv(
                truth_df=truth_df,
                sensor_specs=sensor_specs,
                constraints=constraints,
                cfg=WarmupEnvConfig(
                    state_columns=cfg.state_columns,
                    reward_target_columns=cfg.reward_target_columns,
                    lookback=cfg.lookback,
                    episode_len=cfg.episode_len,
                    seed=int(seed) + rank,
                    base_freq_s=cfg.base_freq_s,
                    event_column=cfg.event_column,
                    normalize_agent_state=cfg.normalize_agent_state,
                    normalization_mean=cfg.normalization_mean,
                    normalization_std=cfg.normalization_std,
                    lambda_warmup_abort=cfg.lambda_warmup_abort,
                    lambda_switch=cfg.lambda_switch,
                    event_reward_multiplier=cfg.event_reward_multiplier,
                    energy_account_enabled=cfg.energy_account_enabled,
                    energy_capacity=cfg.energy_capacity,
                    initial_energy=cfg.initial_energy,
                    harvest_per_step=cfg.harvest_per_step,
                    reserve_energy=cfg.reserve_energy,
                    lambda_energy_deficit=cfg.lambda_energy_deficit,
                    soc_soft_penalty_buffer=cfg.soc_soft_penalty_buffer,
                    lambda_soc_soft_penalty=cfg.lambda_soc_soft_penalty,
                    lambda_duty_balance=cfg.lambda_duty_balance,
                    duty_balance_low=cfg.duty_balance_low,
                    duty_balance_high=cfg.duty_balance_high,
                    duty_balance_grace_steps=cfg.duty_balance_grace_steps,
                    duty_score_feedback=cfg.duty_score_feedback,
                    duty_score_target=cfg.duty_score_target,
                ),
                oracle=oracle,
                random_start=True,
                seed=int(seed) + rank,
                active_score_bonus=float(active_score_bonus),
                warming_score_bonus=float(warming_score_bonus),
                event_start_prob=float(event_start_prob),
                sensor_score_prior=sensor_score_prior,
                event_sensor_score_prior=event_sensor_score_prior,
                non_event_sensor_score_prior=non_event_sensor_score_prior,
                action_scale=float(action_scale),
                candidate_masks=candidate_masks,
            )

        return _factory

    env_fns = [make_one(i) for i in range(int(n_envs))]
    if int(n_envs) <= 1 or str(vec_type).lower() == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="fork")


def train_projected_ppo(
    *,
    truth_df: pd.DataFrame,
    sensor_specs: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle,
    total_timesteps: int,
    n_envs: int = 8,
    seed: int = 42,
    device: str = "auto",
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 1024,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.001,
    clip_range: float = 0.2,
    vec_type: str = "subproc",
    verbose: int = 1,
    tensorboard_log: str | None = None,
    diagnostic_csv: str | None = None,
    diagnostic_freq: int = 0,
    diagnostic_steps: int = 128,
    best_model_path: str | None = None,
    active_score_bonus: float = 0.08,
    warming_score_bonus: float = 0.16,
    event_start_prob: float = 0.35,
    sensor_score_prior: np.ndarray | None = None,
    event_sensor_score_prior: np.ndarray | None = None,
    non_event_sensor_score_prior: np.ndarray | None = None,
    action_scale: float = 0.35,
    candidate_masks: np.ndarray | None = None,
    bc_warmstart_steps: int = 0,
    bc_dataset_steps: int = 2048,
    bc_rollouts: int = 4,
    bc_batch_size: int = 256,
    bc_learning_rate: float = 1e-4,
    bc_event_fraction: float = 0.5,
    bc_greedy_lookahead_steps: int = 4,
    bc_log_path: str | None = None,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    vec_env = make_vec_env(
        truth_df=truth_df,
        sensor_specs=sensor_specs,
        constraints=constraints,
        cfg=cfg,
        oracle=oracle,
        n_envs=int(n_envs),
        seed=int(seed),
        vec_type=vec_type,
        active_score_bonus=float(active_score_bonus),
        warming_score_bonus=float(warming_score_bonus),
        event_start_prob=float(event_start_prob),
        sensor_score_prior=sensor_score_prior,
        event_sensor_score_prior=event_sensor_score_prior,
        non_event_sensor_score_prior=non_event_sensor_score_prior,
        action_scale=float(action_scale),
        candidate_masks=candidate_masks,
    )
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=float(learning_rate),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
            clip_range=float(clip_range),
            ent_coef=float(ent_coef),
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=int(seed),
            device=str(device),
            verbose=int(verbose),
            tensorboard_log=tensorboard_log,
        )
        bc_info: dict[str, float | int | str] = {"enabled": 0}
        if candidate_masks is not None and int(bc_warmstart_steps) > 0:
            observations, actions = collect_oracle_greedy_bc_dataset(
                truth_df=truth_df,
                sensor_specs=sensor_specs,
                constraints=constraints,
                cfg=cfg,
                oracle=oracle,
                candidate_masks=candidate_masks,
                total_steps=int(bc_dataset_steps),
                n_rollouts=int(bc_rollouts),
                event_fraction=float(bc_event_fraction),
                greedy_lookahead_steps=int(bc_greedy_lookahead_steps),
                seed=int(seed) + 50_000,
            )
            bc_info = warmstart_discrete_ppo_policy(
                model,
                observations=observations,
                actions=actions,
                train_steps=int(bc_warmstart_steps),
                batch_size=int(bc_batch_size),
                learning_rate=float(bc_learning_rate),
                seed=int(seed) + 60_000,
            )
            bc_info.update(
                {
                    "enabled": 1,
                    "dataset_steps_requested": int(bc_dataset_steps),
                    "rollouts": int(bc_rollouts),
                    "event_fraction": float(bc_event_fraction),
                    "greedy_lookahead_steps": int(bc_greedy_lookahead_steps),
                }
            )
            if bc_log_path:
                out = Path(bc_log_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(bc_info, indent=2), encoding="utf-8")
        setattr(model, "bc_warmstart_info", bc_info)
        callback = None
        if diagnostic_csv and int(diagnostic_freq) > 0:
            callback = make_ppo_training_diagnostic_callback(
                BaseCallback=BaseCallback,
                csv_path=diagnostic_csv,
                truth_df=truth_df,
                sensor_specs=sensor_specs,
                constraints=constraints,
                cfg=cfg,
                oracle=oracle,
                eval_steps=int(diagnostic_steps),
                eval_freq=int(diagnostic_freq),
                best_model_path=best_model_path,
                active_score_bonus=float(active_score_bonus),
                warming_score_bonus=float(warming_score_bonus),
                sensor_score_prior=sensor_score_prior,
                event_sensor_score_prior=event_sensor_score_prior,
                non_event_sensor_score_prior=non_event_sensor_score_prior,
                action_scale=float(action_scale),
                candidate_masks=candidate_masks,
            )
        if int(total_timesteps) > 0:
            model.learn(total_timesteps=int(total_timesteps), progress_bar=False, callback=callback)
        if callback is not None and getattr(callback, "best_model_saved", False) and best_model_path:
            model = PPO.load(str(best_model_path), device=str(device))
            setattr(model, "bc_warmstart_info", bc_info)
    finally:
        vec_env.close()
    return model


def collect_oracle_greedy_bc_dataset(
    *,
    truth_df: pd.DataFrame,
    sensor_specs: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle,
    candidate_masks: np.ndarray,
    total_steps: int,
    n_rollouts: int,
    event_fraction: float,
    greedy_lookahead_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(sensor_specs))
    if masks.shape[0] == 0:
        raise ValueError("candidate_masks must contain at least one candidate")
    total_steps = max(1, int(total_steps))
    n_rollouts = max(1, int(n_rollouts))
    per_rollout_steps = max(1, int(np.ceil(float(total_steps) / float(n_rollouts))))
    starts = _select_bc_start_indices(
        truth_df,
        steps=per_rollout_steps,
        horizon=int(oracle.cfg.horizon),
        n_rollouts=n_rollouts,
        event_fraction=float(event_fraction),
        seed=int(seed),
        event_column=cfg.event_column,
    )
    obs_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    for rollout_idx, start_idx in enumerate(starts):
        env = WarmupSchedulingEnv(
            truth_df,
            sensor_specs,
            constraints,
            WarmupEnvConfig(
                state_columns=cfg.state_columns,
                reward_target_columns=cfg.reward_target_columns,
                lookback=cfg.lookback,
                episode_len=per_rollout_steps,
                seed=int(cfg.seed) + int(seed) + rollout_idx,
                base_freq_s=cfg.base_freq_s,
                event_column=cfg.event_column,
                normalize_agent_state=cfg.normalize_agent_state,
                normalization_mean=cfg.normalization_mean,
                normalization_std=cfg.normalization_std,
                lambda_warmup_abort=cfg.lambda_warmup_abort,
                lambda_switch=cfg.lambda_switch,
                event_reward_multiplier=cfg.event_reward_multiplier,
                energy_account_enabled=cfg.energy_account_enabled,
                energy_capacity=cfg.energy_capacity,
                initial_energy=cfg.initial_energy,
                harvest_per_step=cfg.harvest_per_step,
                reserve_energy=cfg.reserve_energy,
                lambda_energy_deficit=cfg.lambda_energy_deficit,
                soc_soft_penalty_buffer=cfg.soc_soft_penalty_buffer,
                lambda_soc_soft_penalty=cfg.lambda_soc_soft_penalty,
                lambda_duty_balance=cfg.lambda_duty_balance,
                duty_balance_low=cfg.duty_balance_low,
                duty_balance_high=cfg.duty_balance_high,
                duty_balance_grace_steps=cfg.duty_balance_grace_steps,
                duty_score_feedback=cfg.duty_score_feedback,
                duty_score_target=cfg.duty_score_target,
            ),
            oracle=oracle,
        )
        env.reset(start_idx=int(start_idx))
        for _ in range(per_rollout_steps):
            if len(obs_rows) >= total_steps:
                break
            obs = env._state().astype(np.float32)
            best_action = _oracle_greedy_candidate_index(
                env,
                masks,
                lookahead_steps=int(greedy_lookahead_steps),
            )
            obs_rows.append(obs)
            action_rows.append(int(best_action))
            _, _, done, _ = env.step_mask(masks[int(best_action)])
            if done:
                break
    if not obs_rows:
        raise RuntimeError("No oracle-greedy BC samples were collected")
    return np.vstack(obs_rows).astype(np.float32), np.asarray(action_rows, dtype=np.int64)


def _oracle_greedy_candidate_index(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    lookahead_steps: int,
) -> int:
    snapshot = _snapshot_warmup_env(env)
    best_idx = 0
    best_cost = float("inf")
    for idx, mask in enumerate(candidate_masks):
        _restore_warmup_env(env, snapshot)
        step_costs: list[float] = []
        for _ in range(max(1, int(lookahead_steps))):
            _, _, done, info = env.step_mask(mask)
            oracle_loss = float(info.get("oracle_loss", float("inf")))
            shaping_penalty = float(info.get("shaping_penalty", 0.0))
            cost = oracle_loss + shaping_penalty
            if np.isfinite(cost):
                step_costs.append(float(cost))
            if done:
                break
        cost = float(np.mean(step_costs)) if step_costs else float("inf")
        if np.isfinite(cost) and cost < best_cost:
            best_cost = float(cost)
            best_idx = int(idx)
    _restore_warmup_env(env, snapshot)
    return int(best_idx)


def _snapshot_warmup_env(env: WarmupSchedulingEnv) -> dict[str, object]:
    runtime_state = {
        sensor_id: {
            "mode": runtime.mode,
            "warm_remaining": int(runtime.warm_remaining),
            "last_observed_step": runtime.last_observed_step,
            "warmup_abort_count": int(runtime.warmup_abort_count),
        }
        for sensor_id, runtime in env.runtimes.items()
    }
    return {
        "episode_start_idx": int(env.episode_start_idx),
        "episode_end_idx": int(env.episode_end_idx),
        "current_idx": int(env.current_idx),
        "last_observation": np.array(env.last_observation, copy=True),
        "observed_mask": np.array(env.observed_mask, copy=True),
        "history": np.array(env.history, copy=True),
        "mask_history": np.array(env.mask_history, copy=True),
        "previous_action_mask": np.array(env.previous_action_mask, copy=True),
        "sensor_on_counts": np.array(env.sensor_on_counts, copy=True),
        "elapsed_steps": int(env.elapsed_steps),
        "last_info": copy.deepcopy(env.last_info),
        "rng_state": copy.deepcopy(env.rng.bit_generator.state),
        "runtime_state": runtime_state,
    }


def _restore_warmup_env(env: WarmupSchedulingEnv, snapshot: dict[str, object]) -> None:
    env.episode_start_idx = int(snapshot["episode_start_idx"])
    env.episode_end_idx = int(snapshot["episode_end_idx"])
    env.current_idx = int(snapshot["current_idx"])
    env.last_observation = np.asarray(snapshot["last_observation"], dtype=float).copy()
    env.observed_mask = np.asarray(snapshot["observed_mask"], dtype=float).copy()
    env.history = np.asarray(snapshot["history"], dtype=float).copy()
    env.mask_history = np.asarray(snapshot["mask_history"], dtype=float).copy()
    env.previous_action_mask = np.asarray(snapshot["previous_action_mask"], dtype=float).copy()
    env.sensor_on_counts = np.asarray(snapshot.get("sensor_on_counts", np.zeros(len(env.sensor_specs))), dtype=float).copy()
    env.elapsed_steps = int(snapshot.get("elapsed_steps", 0))
    env.last_info = copy.deepcopy(snapshot["last_info"])
    env.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
    runtime_state = snapshot["runtime_state"]
    assert isinstance(runtime_state, dict)
    for sensor_id, state in runtime_state.items():
        runtime = env.runtimes[str(sensor_id)]
        assert isinstance(state, dict)
        runtime.mode = state["mode"]
        runtime.warm_remaining = int(state["warm_remaining"])
        runtime.last_observed_step = state["last_observed_step"]
        runtime.warmup_abort_count = int(state["warmup_abort_count"])


def _select_bc_start_indices(
    truth_df: pd.DataFrame,
    *,
    steps: int,
    horizon: int,
    n_rollouts: int,
    event_fraction: float,
    seed: int,
    event_column: str,
) -> tuple[int, ...]:
    max_start = max(0, len(truth_df) - int(steps) - int(horizon) - 1)
    if max_start <= 0 or int(n_rollouts) <= 1:
        return (0,)
    rng = np.random.default_rng(int(seed))
    starts: list[int] = []
    n_event = int(round(float(np.clip(event_fraction, 0.0, 1.0)) * int(n_rollouts)))
    event_flags = (
        truth_df[event_column].astype(bool).to_numpy()
        if event_column in truth_df.columns
        else np.zeros(len(truth_df), dtype=bool)
    )
    event_indices = np.flatnonzero(event_flags[: max_start + int(steps)])
    for _ in range(min(n_event, int(n_rollouts))):
        if event_indices.size == 0:
            break
        event_idx = int(rng.choice(event_indices))
        starts.append(int(np.clip(event_idx - int(steps) // 3, 0, max_start)))
    while len(starts) < int(n_rollouts):
        starts.append(int(rng.integers(0, max_start + 1)))
    return tuple(starts)


def warmstart_discrete_ppo_policy(
    model: object,
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float | int | str]:
    if int(train_steps) <= 0:
        return {"enabled": 0}
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PPO BC warm-start requires torch") from exc

    obs = np.asarray(observations, dtype=np.float32)
    act = np.asarray(actions, dtype=np.int64).reshape(-1)
    if obs.shape[0] != act.shape[0]:
        raise ValueError(f"BC observation/action mismatch: {obs.shape[0]} != {act.shape[0]}")
    if obs.shape[0] == 0:
        raise ValueError("BC dataset is empty")

    rng = np.random.default_rng(int(seed))
    policy = model.policy
    device = policy.device
    old_lrs = [float(group.get("lr", learning_rate)) for group in policy.optimizer.param_groups]
    for group in policy.optimizer.param_groups:
        group["lr"] = float(learning_rate)
    losses: list[float] = []
    policy.set_training_mode(True)
    for _ in range(int(train_steps)):
        batch_idx = rng.choice(obs.shape[0], size=min(int(batch_size), obs.shape[0]), replace=obs.shape[0] < int(batch_size))
        obs_tensor = torch.as_tensor(obs[batch_idx], dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(act[batch_idx], dtype=torch.long, device=device)
        distribution = policy.get_distribution(obs_tensor)
        log_prob = distribution.log_prob(action_tensor)
        loss = -log_prob.mean()
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        policy.optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    for group, lr in zip(policy.optimizer.param_groups, old_lrs, strict=False):
        group["lr"] = lr
    policy.set_training_mode(False)
    action_counts = np.bincount(act, minlength=int(np.max(act)) + 1 if act.size else 0)
    return {
        "enabled": 1,
        "samples": int(obs.shape[0]),
        "train_steps": int(train_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "loss_initial": float(losses[0]) if losses else float("nan"),
        "loss_final": float(losses[-1]) if losses else float("nan"),
        "unique_actions": int(np.count_nonzero(action_counts)),
        "dominant_action_rate": float(np.max(action_counts) / np.sum(action_counts)) if action_counts.size else 0.0,
    }


def make_ppo_training_diagnostic_callback(
    *,
    BaseCallback: type,
    csv_path: str,
    truth_df: pd.DataFrame,
    sensor_specs: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle,
    eval_steps: int,
    eval_freq: int,
    best_model_path: str | None,
    active_score_bonus: float,
    warming_score_bonus: float,
    sensor_score_prior: np.ndarray | None,
    event_sensor_score_prior: np.ndarray | None,
    non_event_sensor_score_prior: np.ndarray | None,
    action_scale: float,
    candidate_masks: np.ndarray | None,
):
    from v2.rollout import run_policy_rollout

    class _Callback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.rows: list[dict[str, float | int]] = []
            self.next_eval = int(eval_freq)
            self.best_oracle_loss = float("inf")
            self.best_model_saved = False

        def _on_step(self) -> bool:
            if int(self.num_timesteps) > 0 and int(self.num_timesteps) >= int(self.next_eval):
                self._record()
                self.next_eval = int(self.num_timesteps) + int(eval_freq)
            return True

        def _on_rollout_end(self) -> None:
            return

        def _record(self) -> None:
            env = WarmupSchedulingEnv(
                truth_df,
                sensor_specs,
                constraints,
                WarmupEnvConfig(
                    state_columns=cfg.state_columns,
                    reward_target_columns=cfg.reward_target_columns,
                    lookback=cfg.lookback,
                    episode_len=int(eval_steps),
                    seed=int(cfg.seed) + int(self.num_timesteps),
                    base_freq_s=cfg.base_freq_s,
                    event_column=cfg.event_column,
                    normalize_agent_state=cfg.normalize_agent_state,
                    normalization_mean=cfg.normalization_mean,
                    normalization_std=cfg.normalization_std,
                    lambda_warmup_abort=cfg.lambda_warmup_abort,
                    lambda_switch=cfg.lambda_switch,
                    event_reward_multiplier=cfg.event_reward_multiplier,
                    energy_account_enabled=cfg.energy_account_enabled,
                    energy_capacity=cfg.energy_capacity,
                    initial_energy=cfg.initial_energy,
                    harvest_per_step=cfg.harvest_per_step,
                    reserve_energy=cfg.reserve_energy,
                    lambda_energy_deficit=cfg.lambda_energy_deficit,
                    soc_soft_penalty_buffer=cfg.soc_soft_penalty_buffer,
                    lambda_soc_soft_penalty=cfg.lambda_soc_soft_penalty,
                    lambda_duty_balance=cfg.lambda_duty_balance,
                    duty_balance_low=cfg.duty_balance_low,
                    duty_balance_high=cfg.duty_balance_high,
                    duty_balance_grace_steps=cfg.duty_balance_grace_steps,
                    duty_score_feedback=cfg.duty_score_feedback,
                    duty_score_target=cfg.duty_score_target,
                ),
                oracle=oracle,
            )
            result = run_policy_rollout(
                env,
                SB3ProjectedPPOPolicy(
                    model=self.model,
                    active_score_bonus=float(active_score_bonus),
                    warming_score_bonus=float(warming_score_bonus),
                    sensor_score_prior=sensor_score_prior,
                    event_sensor_score_prior=event_sensor_score_prior,
                    non_event_sensor_score_prior=non_event_sensor_score_prior,
                    action_scale=float(action_scale),
                    candidate_masks=candidate_masks,
                ),
                steps=int(eval_steps),
            )
            metrics = rollout_metrics(result)
            scores = result.scores
            row: dict[str, float | int] = {
                "timesteps": int(self.num_timesteps),
                "reward_mean": float(metrics["reward_mean"]),
                "oracle_loss_mean": float(metrics["oracle_loss_mean"]),
                "instant_mae": float(metrics["instant_mae"]),
                "dtw_mean": float(metrics["dtw_mean"]),
                "power_mean": float(metrics["power_mean"]),
                "warmup_abort_count": int(metrics["warmup_abort_count"]),
                "score_abs_mean": float(np.mean(np.abs(scores))) if scores.size else float("nan"),
                "score_std_mean": float(np.mean(np.std(scores, axis=0))) if scores.size else float("nan"),
                "selected_rate_mean": float(np.mean(result.selected_masks)) if result.selected_masks.size else float("nan"),
            }
            self.rows.append(row)
            oracle_loss = float(row["oracle_loss_mean"])
            if np.isfinite(oracle_loss) and oracle_loss < self.best_oracle_loss and best_model_path:
                self.best_oracle_loss = oracle_loss
                out_model = Path(best_model_path)
                out_model.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(out_model))
                self.best_model_saved = True
            out = Path(csv_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.rows).to_csv(out, index=False)

    return _Callback()


def evaluate_sb3_model(
    *,
    model: object,
    truth_df: pd.DataFrame,
    sensor_specs: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle,
    steps: int,
    start_indices: list[int] | tuple[int, ...] | None = None,
    active_score_bonus: float = 0.08,
    warming_score_bonus: float = 0.16,
    sensor_score_prior: np.ndarray | None = None,
    event_sensor_score_prior: np.ndarray | None = None,
    non_event_sensor_score_prior: np.ndarray | None = None,
    action_scale: float = 0.35,
    candidate_masks: np.ndarray | None = None,
) -> tuple[RolloutResult, dict[str, float | str | int]]:
    from v2.rollout import concat_rollout_results, run_policy_rollout

    policy = SB3ProjectedPPOPolicy(
        model=model,
        active_score_bonus=float(active_score_bonus),
        warming_score_bonus=float(warming_score_bonus),
        sensor_score_prior=sensor_score_prior,
        event_sensor_score_prior=event_sensor_score_prior,
        non_event_sensor_score_prior=non_event_sensor_score_prior,
        action_scale=float(action_scale),
        candidate_masks=candidate_masks,
    )
    starts = tuple(int(x) for x in (start_indices or (0,)))
    rollouts = []
    for offset, start_idx in enumerate(starts):
        env = WarmupSchedulingEnv(
            truth_df,
            sensor_specs,
            constraints,
            WarmupEnvConfig(
                state_columns=cfg.state_columns,
                reward_target_columns=cfg.reward_target_columns,
                lookback=cfg.lookback,
                episode_len=cfg.episode_len,
                seed=int(cfg.seed) + int(offset),
                base_freq_s=cfg.base_freq_s,
                event_column=cfg.event_column,
                normalize_agent_state=cfg.normalize_agent_state,
                normalization_mean=cfg.normalization_mean,
                normalization_std=cfg.normalization_std,
                lambda_warmup_abort=cfg.lambda_warmup_abort,
                lambda_switch=cfg.lambda_switch,
                event_reward_multiplier=cfg.event_reward_multiplier,
                energy_account_enabled=cfg.energy_account_enabled,
                energy_capacity=cfg.energy_capacity,
                initial_energy=cfg.initial_energy,
                harvest_per_step=cfg.harvest_per_step,
                reserve_energy=cfg.reserve_energy,
                lambda_energy_deficit=cfg.lambda_energy_deficit,
                soc_soft_penalty_buffer=cfg.soc_soft_penalty_buffer,
                lambda_soc_soft_penalty=cfg.lambda_soc_soft_penalty,
                lambda_duty_balance=cfg.lambda_duty_balance,
                duty_balance_low=cfg.duty_balance_low,
                duty_balance_high=cfg.duty_balance_high,
                duty_balance_grace_steps=cfg.duty_balance_grace_steps,
                duty_score_feedback=cfg.duty_score_feedback,
                duty_score_target=cfg.duty_score_target,
            ),
            oracle=oracle,
        )
        rollouts.append(run_policy_rollout(env, policy, steps=int(steps), start_idx=int(start_idx)))
    result = rollouts[0] if len(rollouts) == 1 else concat_rollout_results(rollouts, policy_name=policy.name)
    return result, rollout_metrics(result)


def save_model(model: object, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
