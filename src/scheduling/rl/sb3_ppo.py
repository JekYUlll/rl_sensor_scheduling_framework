from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from estimators.state_summary import flatten_rl_state
from evaluation.constraint_metrics import summarize_constraint_metrics
from evaluation.cost_metrics import compute_step_cost
from reward.forecast_reward import FrozenForecastRewardOracle
from scheduling.online_projector import OnlineSubsetProjector


DEFAULT_TRAINING_LOG_COLUMNS = [
    "reward",
    "trace_P",
    "uncertainty",
    "forecast_reward",
    "power",
    "peak_power_max",
    "total_energy",
    "peak_violation_rate",
    "coverage",
    "lambda_avg",
    "lambda_energy",
    "avg_power_violation",
    "energy_violation",
    "val_objective",
    "val_reward",
    "val_forecast_reward",
    "val_power",
    "val_peak_violation_rate",
]


def _score_ranking(action: np.ndarray, sensor_ids: list[str]) -> list[str]:
    arr = np.asarray(action, dtype=float).reshape(-1)
    pairs = sorted(zip(sensor_ids, arr, strict=True), key=lambda item: float(item[1]), reverse=True)
    return [sid for sid, _ in pairs]


@dataclass
class PPOPolicyAdapter:
    sensor_ids: list[str]
    cfg: dict
    selector: OnlineSubsetProjector
    model: PPO | None = None

    def __post_init__(self) -> None:
        self.device = str(self.cfg.get("device", "auto"))

    def act(self, state_vec: np.ndarray, greedy: bool = False, prev_selected: list[str] | None = None) -> list[str]:
        if self.model is None:
            raise RuntimeError("PPO model is not loaded")
        action, _ = self.model.predict(np.asarray(state_vec, dtype=np.float32), deterministic=bool(greedy))
        return _score_ranking(np.asarray(action, dtype=float), self.sensor_ids)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("PPO model is not initialized")
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path, device=self.device)


class WindblownSubsetGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env,
        estimator,
        selector: OnlineSubsetProjector,
        task_cost_cfg: dict,
        state_columns: list[str],
        reward_target_indices: list[int],
        constraint_budgets: dict[str, float | None],
        reward_oracle: FrozenForecastRewardOracle | None = None,
    ) -> None:
        super().__init__()
        self.env = env
        self.estimator = estimator
        self.selector = selector
        self.task_cost_cfg = dict(task_cost_cfg)
        self.state_columns = list(state_columns)
        self.reward_target_indices = list(reward_target_indices)
        self.constraint_budgets = dict(constraint_budgets)
        self.reward_oracle = reward_oracle
        self.sensor_ids = list(selector.sensor_ids)
        self.col_to_idx = {name: i for i, name in enumerate(self.state_columns)}
        state_dim = len(flatten_rl_state({**self.estimator.get_rl_state_features(), "event": False}))
        self.observation_space = spaces.Box(low=-1.0e9, high=1.0e9, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.sensor_ids),), dtype=np.float32)
        self.current_event = False
        self.prev_selected: list[str] = []
        self.estimate_history: list[np.ndarray] = []
        self.observed_mask_history: list[np.ndarray] = []
        self.total_reward = 0.0
        self.trace_hist: list[float] = []
        self.uncertainty_hist: list[float] = []
        self.forecast_hist: list[float] = []
        self.power_hist: list[float] = []
        self.peak_power_hist: list[float] = []
        self.startup_extra_hist: list[float] = []
        self.coverage_hist: list[float] = []

    def _current_obs(self) -> np.ndarray:
        state = self.estimator.get_rl_state_features()
        state["event"] = self.current_event
        return np.asarray(flatten_rl_state(state), dtype=np.float32)

    def _prediction_error(self, latent_state: dict[str, float]) -> float:
        if not self.reward_target_indices:
            return 0.0
        truth = np.asarray([float(latent_state[col]) for col in self.state_columns], dtype=float)
        return self.estimator.normalized_state_error(truth, dims=self.reward_target_indices)

    def _forecast_error(self) -> float:
        oracle = self.reward_oracle
        if oracle is None:
            return 0.0
        future_truth = self.env.peek_future_targets(oracle.horizon, list(self.state_columns))
        if not oracle.ready(len(self.estimate_history), len(future_truth)):
            return 0.0
        history = np.asarray(self.estimate_history[-oracle.lookback :], dtype=float)
        mask_history = np.asarray(self.observed_mask_history[-oracle.lookback :], dtype=float)
        return oracle.score(history, future_truth, mask_history)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        reset_out = self.env.reset()
        self.estimator.reset()
        self.current_event = bool(reset_out.get("event_flags", {}).get("event", False))
        self.prev_selected = []
        self.estimate_history = [self.estimator.get_state_estimate().copy()]
        self.observed_mask_history = [np.ones(len(self.state_columns), dtype=float)]
        self.total_reward = 0.0
        self.trace_hist = []
        self.uncertainty_hist = []
        self.forecast_hist = []
        self.power_hist = []
        self.peak_power_hist = []
        self.startup_extra_hist = []
        self.coverage_hist = []
        return self._current_obs(), {}

    def step(self, action: np.ndarray):
        ranked = _score_ranking(np.asarray(action, dtype=float), self.sensor_ids)
        selected = self.selector.project_ranked(ranked, prev_selected=self.prev_selected)

        power_info = self.selector.power_metrics(selected, prev_selected=self.prev_selected)
        step = self.env.step(selected)
        self.estimator.predict()
        self.estimator.update(step["available_observations"])
        observed_sensor_ids = [str(obs["sensor_id"]) for obs in step["available_observations"] if obs.get("available", False)]
        power_cost = float(power_info["steady_power"])
        power_ratio = power_cost / max(float(getattr(self.selector, "per_step_budget", power_cost or 1.0)), 1e-6)
        self.estimator.on_step(
            selected_sensor_ids=selected,
            power_ratio=power_ratio,
            observed_sensor_ids=observed_sensor_ids,
        )

        self.current_event = bool(step.get("event_flags", {}).get("event", False))
        unc_summary = self.estimator.get_uncertainty_summary()
        trace_p = float(unc_summary["trace_P"])
        self.estimate_history.append(self.estimator.get_state_estimate().copy())
        mask = np.zeros(len(self.state_columns), dtype=float)
        for obs in step["available_observations"]:
            for var_name in obs.get("variables", []):
                idx = self.col_to_idx.get(var_name)
                if idx is not None:
                    mask[idx] = 1.0
        self.observed_mask_history.append(mask)
        pred_error = self._prediction_error(step["latent_state"])
        forecast_error = self._forecast_error()
        task_cost = compute_step_cost(
            uncertainty_summary=unc_summary,
            power_cost=power_cost,
            switch_count=len(set(self.prev_selected) ^ set(selected)),
            coverage_ratio=self.estimator.get_rl_state_features().get("coverage_ratio", []),
            cost_cfg=self.task_cost_cfg,
            prediction_error=pred_error,
        )
        task_cost += float(self.task_cost_cfg.get("beta_forecast", 0.0)) * forecast_error
        reward = -float(task_cost)

        self.total_reward += reward
        self.trace_hist.append(trace_p)
        self.uncertainty_hist.append(float(unc_summary.get(str(self.task_cost_cfg.get("uncertainty_metric", "trace_P")), trace_p)))
        self.forecast_hist.append(float(forecast_error))
        self.power_hist.append(power_cost)
        self.peak_power_hist.append(float(power_info["peak_power"]))
        self.startup_extra_hist.append(float(power_info["startup_extra_power"]))
        cov = self.estimator.get_rl_state_features().get("coverage_ratio", [])
        self.coverage_hist.append(float(np.mean(cov)) if cov else 0.0)
        self.prev_selected = list(selected)

        terminated = bool(step["done"])
        info: dict[str, float | dict] = {}
        if terminated:
            constraint_metrics = summarize_constraint_metrics(
                steady_power_hist=self.power_hist,
                peak_power_hist=self.peak_power_hist,
                startup_extra_hist=self.startup_extra_hist,
                average_power_budget=self.constraint_budgets.get("average_power_budget"),
                episode_energy_budget=self.constraint_budgets.get("episode_energy_budget"),
                peak_power_budget=self.constraint_budgets.get("peak_power_budget"),
            )
            info["episode_summary"] = {
                "reward": float(self.total_reward),
                "trace_P": float(np.mean(self.trace_hist)) if self.trace_hist else float("nan"),
                "uncertainty": float(np.mean(self.uncertainty_hist)) if self.uncertainty_hist else float("nan"),
                "forecast_reward": float(np.mean(self.forecast_hist)) if self.forecast_hist else float("nan"),
                "power": float(constraint_metrics["power_mean"]),
                "peak_power_max": float(constraint_metrics["peak_power_max"]),
                "total_energy": float(constraint_metrics["total_energy"]),
                "peak_violation_rate": float(constraint_metrics["peak_violation_rate"]),
                "coverage": float(np.mean(self.coverage_hist)) if self.coverage_hist else float("nan"),
                "avg_power_violation": float(constraint_metrics["avg_power_violation"]),
                "energy_violation": float(constraint_metrics["energy_violation"]),
            }
        return self._current_obs(), reward, terminated, False, info


class PPOTrainingCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_dir: Path,
        eval_env_factory: Callable[[], WindblownSubsetGymEnv],
        eval_interval_episodes: int,
        eval_episodes: int,
        best_model_path: Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = Path(run_dir)
        self.eval_env_factory = eval_env_factory
        self.eval_interval_episodes = max(1, int(eval_interval_episodes))
        self.eval_episodes = max(1, int(eval_episodes))
        self.best_model_path = Path(best_model_path)
        self.episode_rows: list[dict[str, float]] = []
        self.best_objective = float("-inf")

    def _evaluate(self) -> dict[str, float]:
        rewards: list[float] = []
        powers: list[float] = []
        forecasts: list[float] = []
        peaks: list[float] = []
        eval_env = self.eval_env_factory()
        for _ in range(self.eval_episodes):
            obs, _ = eval_env.reset()
            terminated = False
            while not terminated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, _, info = eval_env.step(action)
            summary = info["episode_summary"]
            rewards.append(float(summary["reward"]))
            powers.append(float(summary["power"]))
            forecasts.append(float(summary["forecast_reward"]))
            peaks.append(float(summary["peak_violation_rate"]))
        return {
            "val_objective": float(np.mean(rewards)),
            "val_reward": float(np.mean(rewards)),
            "val_forecast_reward": float(np.mean(forecasts)),
            "val_power": float(np.mean(powers)),
            "val_peak_violation_rate": float(np.mean(peaks)),
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos, strict=False):
            if not done:
                continue
            summary = info.get("episode_summary")
            if not summary:
                continue
            row = {key: float("nan") for key in DEFAULT_TRAINING_LOG_COLUMNS}
            row.update(
                {
                    "reward": float(summary["reward"]),
                    "trace_P": float(summary["trace_P"]),
                    "uncertainty": float(summary["uncertainty"]),
                    "forecast_reward": float(summary["forecast_reward"]),
                    "power": float(summary["power"]),
                    "peak_power_max": float(summary["peak_power_max"]),
                    "total_energy": float(summary["total_energy"]),
                    "peak_violation_rate": float(summary["peak_violation_rate"]),
                    "coverage": float(summary["coverage"]),
                    "avg_power_violation": float(summary["avg_power_violation"]),
                    "energy_violation": float(summary["energy_violation"]),
                    "lambda_avg": 0.0,
                    "lambda_energy": 0.0,
                }
            )
            self.episode_rows.append(row)
            if len(self.episode_rows) % self.eval_interval_episodes == 0:
                eval_metrics = self._evaluate()
                self.episode_rows[-1].update(eval_metrics)
                if eval_metrics["val_objective"] > self.best_objective:
                    self.best_objective = float(eval_metrics["val_objective"])
                    self.model.save(str(self.best_model_path))
        return True

    def export_training_log(self) -> pd.DataFrame:
        df = pd.DataFrame(self.episode_rows, columns=DEFAULT_TRAINING_LOG_COLUMNS)
        path = self.run_dir / "training_log.csv"
        df.to_csv(path, index=False)
        return df


def build_ppo_model(cfg: dict, env: WindblownSubsetGymEnv, device: str = "auto") -> PPO:
    ppo_cfg = dict(cfg.get("ppo", {}))
    hidden_dims = list(ppo_cfg.get("policy_hidden_dims", cfg.get("network", {}).get("hidden_dims", [128, 128])))
    policy_kwargs = {"net_arch": dict(pi=hidden_dims, vf=hidden_dims)}
    vec_env = DummyVecEnv([lambda: env])
    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 256)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=int(ppo_cfg.get("verbose", 0)),
    )
