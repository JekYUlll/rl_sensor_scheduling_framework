from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, cast
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv, sync_envs_normalization
from estimators.state_summary import flatten_rl_state
from evaluation.constraint_metrics import summarize_constraint_metrics
from reward.forecast_reward import FrozenForecastRewardEnsemble, FrozenForecastRewardOracle
from reward.mainline_reward import compute_forecast_task_terms
from scheduling.online_projector import OnlineSubsetProjector
DEFAULT_TRAINING_LOG_COLUMNS = ['reward', 'task_reward', 'task_loss', 'trace_P', 'uncertainty', 'forecast_loss', 'switch_penalty', 'coverage_penalty', 'constraint_violation', 'state_tracking_loss', 'power', 'peak_power_max', 'total_energy', 'peak_violation_rate', 'coverage', 'lambda_avg', 'lambda_energy', 'avg_power_violation', 'energy_violation', 'val_objective', 'val_reward', 'val_forecast_loss', 'val_power', 'val_peak_violation_rate']

def _score_ranking(action: np.ndarray, sensor_ids: list[str]) -> list[str]:
    arr = np.asarray(action, dtype=float).reshape(-1)
    pairs = sorted(zip(sensor_ids, arr, strict=True), key=lambda item: float(item[1]), reverse=True)
    return [sid for sid, _ in pairs]

def _temporal_features(env) -> dict[str, float]:
    absolute_time_index = env.get_absolute_time_index() if hasattr(env, 'get_absolute_time_index') else env.get_time_index()
    base_freq_s = int(getattr(getattr(env, 'cfg', None), 'base_freq_s', 1))
    seconds = float(absolute_time_index) * float(base_freq_s)
    phase = (seconds % 86400.0) / 86400.0
    theta = 2.0 * np.pi * phase
    return {
        'time_of_day_sin': float(np.sin(theta)),
        'time_of_day_cos': float(np.cos(theta)),
    }

@dataclass
class PPOPolicyAdapter:
    sensor_ids: list[str]
    cfg: dict
    selector: OnlineSubsetProjector
    model: PPO | None = None
    obs_mean: np.ndarray | None = None
    obs_var: np.ndarray | None = None
    obs_epsilon: float = 1e-08
    clip_obs: float = 10.0

    def __post_init__(self) -> None:
        self.device = str(self.cfg.get('device', 'auto'))

    def _normalize_obs(self, state_vec: np.ndarray) -> np.ndarray:
        obs = np.asarray(state_vec, dtype=np.float32)
        if self.obs_mean is None or self.obs_var is None:
            return obs
        denom = np.sqrt(np.maximum(self.obs_var, 0.0) + float(self.obs_epsilon))
        norm = (obs - self.obs_mean.astype(np.float32)) / denom.astype(np.float32)
        return np.clip(norm, -float(self.clip_obs), float(self.clip_obs)).astype(np.float32)

    def act(self, state_vec: np.ndarray, greedy: bool=False, prev_selected: list[str] | None=None) -> list[str]:
        if self.model is None:
            raise RuntimeError('PPO model is not loaded')
        obs = self._normalize_obs(np.asarray(state_vec, dtype=np.float32))
        action, _ = self.model.predict(obs, deterministic=bool(greedy))
        return _score_ranking(np.asarray(action, dtype=float), self.sensor_ids)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError('PPO model is not initialized')
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path, device=self.device)
        stats_path = Path(path).with_suffix('.norm.npz')
        if stats_path.exists():
            stats = np.load(stats_path)
            self.obs_mean = np.asarray(stats['mean'], dtype=np.float32)
            self.obs_var = np.asarray(stats['var'], dtype=np.float32)
            epsilon_arr = np.asarray(stats['epsilon'], dtype=np.float32).reshape(-1)
            clip_obs_arr = np.asarray(stats['clip_obs'], dtype=np.float32).reshape(-1)
            if epsilon_arr.size > 0:
                self.obs_epsilon = float(epsilon_arr[0])
            if clip_obs_arr.size > 0:
                self.clip_obs = float(clip_obs_arr[0])

def save_obs_normalization_stats(vec_env: VecNormalize, path: Path) -> None:
    if vec_env.obs_rms is None:
        return
    obs_rms = cast(Any, vec_env.obs_rms)
    np.savez(path, mean=np.asarray(obs_rms.mean, dtype=np.float32), var=np.asarray(obs_rms.var, dtype=np.float32), epsilon=np.asarray([float(vec_env.epsilon)], dtype=np.float32), clip_obs=np.asarray([float(vec_env.clip_obs)], dtype=np.float32))

def make_vecnormalize(vec_env: DummyVecEnv, *, gamma: float, norm_obs: bool, norm_reward: bool, clip_obs: float, clip_reward: float, training: bool=True) -> VecNormalize:
    return VecNormalize(vec_env, training=training, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=clip_obs, clip_reward=clip_reward, gamma=gamma)

class WindblownSubsetGymEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, *, env, estimator, selector: OnlineSubsetProjector, task_reward_cfg: dict, state_columns: list[str], reward_target_indices: list[int], constraint_budgets: dict[str, float | None], reward_oracle: FrozenForecastRewardOracle | FrozenForecastRewardEnsemble | None=None) -> None:
        super().__init__()
        self.env = env
        self.estimator = estimator
        self.selector = selector
        self.task_reward_cfg = dict(task_reward_cfg)
        self.state_columns = list(state_columns)
        self.reward_target_indices = list(reward_target_indices)
        self.constraint_budgets = dict(constraint_budgets)
        self.reward_oracle = reward_oracle
        self.sensor_ids = list(selector.sensor_ids)
        self.col_to_idx = {name: i for i, name in enumerate(self.state_columns)}
        init_state = self.estimator.get_rl_state_features()
        init_state['event'] = False
        init_state.update(_temporal_features(self.env))
        state_dim = len(flatten_rl_state(init_state))
        self.observation_space = spaces.Box(low=-1000000000.0, high=1000000000.0, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.sensor_ids),), dtype=np.float32)
        self.current_event = False
        self.prev_selected: list[str] = []
        self.estimate_history: list[np.ndarray] = []
        self.observed_mask_history: list[np.ndarray] = []
        self.time_index_history: list[int] = []
        self.trace_context_history: list[float] = []
        self.power_context_history: list[float] = []
        self.peak_power_context_history: list[float] = []
        self.event_context_history: list[int] = []
        self.total_reward = 0.0
        self.trace_hist: list[float] = []
        self.uncertainty_hist: list[float] = []
        self.forecast_hist: list[float] = []
        self.task_loss_hist: list[float] = []
        self.switch_penalty_hist: list[float] = []
        self.coverage_penalty_hist: list[float] = []
        self.violation_penalty_hist: list[float] = []
        self.state_tracking_hist: list[float] = []
        self.power_hist: list[float] = []
        self.peak_power_hist: list[float] = []
        self.startup_extra_hist: list[float] = []
        self.coverage_hist: list[float] = []

    def _current_obs(self) -> np.ndarray:
        state = self.estimator.get_rl_state_features()
        state['event'] = self.current_event
        state.update(_temporal_features(self.env))
        return np.asarray(flatten_rl_state(state), dtype=np.float32)

    def _state_tracking_loss(self, latent_state: dict[str, float]) -> float:
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
        history = np.asarray(self.estimate_history[-oracle.lookback:], dtype=float)
        mask_history = np.asarray(self.observed_mask_history[-oracle.lookback:], dtype=float)
        time_history = np.asarray(self.time_index_history[-oracle.lookback:], dtype=int)
        context_window = {
            'trace_p': np.asarray(self.trace_context_history[-oracle.lookback:], dtype=float),
            'power': np.asarray(self.power_context_history[-oracle.lookback:], dtype=float),
            'peak_power': np.asarray(self.peak_power_context_history[-oracle.lookback:], dtype=float),
            'event_flags': np.asarray(self.event_context_history[-oracle.lookback:], dtype=float),
        }
        return oracle.score(
            history,
            future_truth,
            mask_history,
            time_history,
            context_series_window=context_window,
        )

    def reset(self, *, seed: int | None=None, options: dict | None=None):
        super().reset(seed=seed)
        reset_out = self.env.reset()
        self.estimator.reset()
        self.current_event = bool(reset_out.get('event_flags', {}).get('event', False))
        self.prev_selected = []
        self.estimate_history = [self.estimator.get_state_estimate().copy()]
        self.observed_mask_history = [np.ones(len(self.state_columns), dtype=float)]
        self.time_index_history = [int(self.env.get_absolute_time_index()) if hasattr(self.env, 'get_absolute_time_index') else int(self.env.get_time_index())]
        initial_trace = float(self.estimator.get_uncertainty_summary()['trace_P'])
        self.trace_context_history = [initial_trace]
        self.power_context_history = [0.0]
        self.peak_power_context_history = [0.0]
        self.event_context_history = [1 if self.current_event else 0]
        self.total_reward = 0.0
        self.trace_hist = []
        self.uncertainty_hist = []
        self.forecast_hist = []
        self.task_loss_hist = []
        self.switch_penalty_hist = []
        self.coverage_penalty_hist = []
        self.violation_penalty_hist = []
        self.state_tracking_hist = []
        self.power_hist = []
        self.peak_power_hist = []
        self.startup_extra_hist = []
        self.coverage_hist = []
        return (self._current_obs(), {})

    def step(self, action: np.ndarray):
        ranked = _score_ranking(np.asarray(action, dtype=float), self.sensor_ids)
        selected = self.selector.project_ranked(ranked, prev_selected=self.prev_selected)
        power_info = self.selector.power_metrics(selected, prev_selected=self.prev_selected)
        step = self.env.step(selected)
        self.estimator.predict()
        self.estimator.update(step['available_observations'])
        observed_sensor_ids = [str(obs['sensor_id']) for obs in step['available_observations'] if obs.get('available', False)]
        power_cost = float(power_info['steady_power'])
        power_ratio = power_cost / max(float(getattr(self.selector, 'per_step_budget', power_cost or 1.0)), 1e-06)
        self.estimator.on_step(selected_sensor_ids=selected, power_ratio=power_ratio, observed_sensor_ids=observed_sensor_ids, sensor_status=step.get('sensor_status'))
        self.current_event = bool(step.get('event_flags', {}).get('event', False))
        unc_summary = self.estimator.get_uncertainty_summary()
        trace_p = float(unc_summary['trace_P'])
        self.estimate_history.append(self.estimator.get_state_estimate().copy())
        mask = np.zeros(len(self.state_columns), dtype=float)
        for obs in step['available_observations']:
            for var_name in obs.get('variables', []):
                idx = self.col_to_idx.get(var_name)
                if idx is not None:
                    mask[idx] = 1.0
        self.observed_mask_history.append(mask)
        self.time_index_history.append(int(self.env.get_absolute_time_index()) if hasattr(self.env, 'get_absolute_time_index') else int(self.env.get_time_index()))
        self.trace_context_history.append(trace_p)
        self.power_context_history.append(power_cost)
        self.peak_power_context_history.append(float(power_info['peak_power']))
        self.event_context_history.append(1 if self.current_event else 0)
        state_tracking_loss = 0.0
        if float(self.task_reward_cfg.get('lambda_state_tracking', 0.0)) > 0.0:
            state_tracking_loss = self._state_tracking_loss(step['latent_state'])
        forecast_loss = self._forecast_error()
        reward_terms = compute_forecast_task_terms(
            forecast_loss=forecast_loss,
            switch_count=len(set(self.prev_selected) ^ set(selected)),
            coverage_ratio=self.estimator.get_rl_state_features().get('coverage_ratio', []),
            steady_power=power_cost,
            peak_power=float(power_info['peak_power']),
            steady_limit=max(float(self.selector.per_step_budget) - float(self.selector.safety_margin), 0.0),
            peak_limit=None if self.selector.startup_peak_budget is None else max(float(self.selector.startup_peak_budget) - float(self.selector.safety_margin), 0.0),
            reward_cfg=self.task_reward_cfg,
            state_tracking_loss=state_tracking_loss,
        )
        reward = float(reward_terms['task_reward'])
        self.total_reward += reward
        self.trace_hist.append(trace_p)
        self.uncertainty_hist.append(float(unc_summary.get('weighted_trace_P_norm', trace_p)))
        self.forecast_hist.append(float(forecast_loss))
        self.task_loss_hist.append(float(reward_terms['task_loss']))
        self.switch_penalty_hist.append(float(reward_terms['switch_penalty_raw']))
        self.coverage_penalty_hist.append(float(reward_terms['coverage_penalty_raw']))
        self.violation_penalty_hist.append(float(reward_terms['violation_penalty_raw']))
        self.state_tracking_hist.append(float(reward_terms['state_tracking_loss']))
        self.power_hist.append(power_cost)
        self.peak_power_hist.append(float(power_info['peak_power']))
        self.startup_extra_hist.append(float(power_info['startup_extra_power']))
        cov = self.estimator.get_rl_state_features().get('coverage_ratio', [])
        self.coverage_hist.append(float(np.mean(cov)) if cov else 0.0)
        self.prev_selected = list(selected)
        terminated = bool(step['done'])
        info: dict[str, float | dict] = {}
        if terminated:
            constraint_metrics = summarize_constraint_metrics(steady_power_hist=self.power_hist, peak_power_hist=self.peak_power_hist, startup_extra_hist=self.startup_extra_hist, average_power_budget=self.constraint_budgets.get('average_power_budget'), episode_energy_budget=self.constraint_budgets.get('episode_energy_budget'), peak_power_budget=self.constraint_budgets.get('peak_power_budget'))
            info['episode_summary'] = {'reward': float(self.total_reward), 'task_reward': float(self.total_reward), 'task_loss': float(np.mean(self.task_loss_hist)) if self.task_loss_hist else float('nan'), 'trace_P': float(np.mean(self.trace_hist)) if self.trace_hist else float('nan'), 'uncertainty': float(np.mean(self.uncertainty_hist)) if self.uncertainty_hist else float('nan'), 'forecast_loss': float(np.mean(self.forecast_hist)) if self.forecast_hist else float('nan'), 'switch_penalty': float(np.mean(self.switch_penalty_hist)) if self.switch_penalty_hist else float('nan'), 'coverage_penalty': float(np.mean(self.coverage_penalty_hist)) if self.coverage_penalty_hist else float('nan'), 'constraint_violation': float(np.mean(self.violation_penalty_hist)) if self.violation_penalty_hist else float('nan'), 'state_tracking_loss': float(np.mean(self.state_tracking_hist)) if self.state_tracking_hist else float('nan'), 'power': float(constraint_metrics['power_mean']), 'peak_power_max': float(constraint_metrics['peak_power_max']), 'total_energy': float(constraint_metrics['total_energy']), 'peak_violation_rate': float(constraint_metrics['peak_violation_rate']), 'coverage': float(np.mean(self.coverage_hist)) if self.coverage_hist else float('nan'), 'avg_power_violation': float(constraint_metrics['avg_power_violation']), 'energy_violation': float(constraint_metrics['energy_violation'])}
        return (self._current_obs(), reward, terminated, False, info)

class PPOTrainingCallback(BaseCallback):

    def __init__(self, *, run_dir: Path, eval_env_factory: Callable[[], VecNormalize], eval_interval_episodes: int, eval_episodes: int, best_model_path: Path, verbose: int=0) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = Path(run_dir)
        self.eval_env_factory = eval_env_factory
        self.eval_interval_episodes = max(1, int(eval_interval_episodes))
        self.eval_episodes = max(1, int(eval_episodes))
        self.best_model_path = Path(best_model_path)
        self.episode_rows: list[dict[str, float]] = []
        self.best_objective = float('-inf')

    def _evaluate(self) -> dict[str, float]:
        rewards: list[float] = []
        powers: list[float] = []
        forecasts: list[float] = []
        peaks: list[float] = []
        eval_env = self.eval_env_factory()
        train_env = self.model.get_env()
        if train_env is None:
            raise RuntimeError('PPO model is missing its training environment')
        sync_envs_normalization(train_env, eval_env)
        for _ in range(self.eval_episodes):
            obs = cast(np.ndarray, eval_env.reset())
            done = False
            info: dict[str, Any] = {}
            while not done:
                action, _ = self.model.predict(cast(np.ndarray, obs), deterministic=True)
                obs, _, dones, infos = eval_env.step(action)
                done = bool(dones[0])
                info = infos[0] if infos else {}
            summary = dict(info.get('episode_summary', {}))
            if not summary:
                continue
            rewards.append(float(summary['reward']))
            powers.append(float(summary['power']))
            forecasts.append(float(summary['forecast_loss']))
            peaks.append(float(summary['peak_violation_rate']))
        eval_env.close()
        if not rewards:
            return {'val_objective': float('-inf'), 'val_reward': float('nan'), 'val_forecast_loss': float('nan'), 'val_power': float('nan'), 'val_peak_violation_rate': float('nan')}
        return {'val_objective': float(np.mean(rewards)), 'val_reward': float(np.mean(rewards)), 'val_forecast_loss': float(np.mean(forecasts)), 'val_power': float(np.mean(powers)), 'val_peak_violation_rate': float(np.mean(peaks))}

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for done, info in zip(dones, infos, strict=False):
            if not done:
                continue
            summary = info.get('episode_summary')
            if not summary:
                continue
            row = {key: float('nan') for key in DEFAULT_TRAINING_LOG_COLUMNS}
            row.update({'reward': float(summary['reward']), 'task_reward': float(summary['task_reward']), 'task_loss': float(summary['task_loss']), 'trace_P': float(summary['trace_P']), 'uncertainty': float(summary['uncertainty']), 'forecast_loss': float(summary['forecast_loss']), 'switch_penalty': float(summary['switch_penalty']), 'coverage_penalty': float(summary['coverage_penalty']), 'constraint_violation': float(summary['constraint_violation']), 'state_tracking_loss': float(summary['state_tracking_loss']), 'power': float(summary['power']), 'peak_power_max': float(summary['peak_power_max']), 'total_energy': float(summary['total_energy']), 'peak_violation_rate': float(summary['peak_violation_rate']), 'coverage': float(summary['coverage']), 'avg_power_violation': float(summary['avg_power_violation']), 'energy_violation': float(summary['energy_violation']), 'lambda_avg': 0.0, 'lambda_energy': 0.0})
            self.episode_rows.append(row)
            if len(self.episode_rows) % self.eval_interval_episodes == 0:
                eval_metrics = self._evaluate()
                self.episode_rows[-1].update(eval_metrics)
                if eval_metrics['val_objective'] > self.best_objective:
                    self.best_objective = float(eval_metrics['val_objective'])
                    self.model.save(str(self.best_model_path))
        return True

    def export_training_log(self) -> pd.DataFrame:
        df = pd.DataFrame.from_records(self.episode_rows)
        if df.empty:
            df = pd.DataFrame({col: pd.Series(dtype=float) for col in DEFAULT_TRAINING_LOG_COLUMNS})
        else:
            df = df.reindex(columns=DEFAULT_TRAINING_LOG_COLUMNS)
        path = self.run_dir / 'training_log.csv'
        df.to_csv(path, index=False)
        return df

def build_ppo_model(cfg: dict, env_fns: Sequence[Callable[[], WindblownSubsetGymEnv]], device: str='auto') -> PPO:
    ppo_cfg = dict(cfg.get('ppo', {}))
    hidden_dims = list(ppo_cfg.get('policy_hidden_dims', cfg.get('network', {}).get('hidden_dims', [128, 128])))
    policy_kwargs = {'net_arch': dict(pi=hidden_dims, vf=hidden_dims)}
    if not env_fns:
        raise ValueError('PPO requires at least one environment factory')
    vec_env = DummyVecEnv(list(env_fns))
    gamma = float(ppo_cfg.get('gamma', 0.99))
    norm_obs = bool(ppo_cfg.get('normalize_observations', True))
    norm_reward = bool(ppo_cfg.get('normalize_reward', True))
    clip_obs = float(ppo_cfg.get('clip_obs', 10.0))
    clip_reward = float(ppo_cfg.get('clip_reward', 10.0))
    vec_env = make_vecnormalize(vec_env, gamma=gamma, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=clip_obs, clip_reward=clip_reward, training=True)
    return PPO(policy='MlpPolicy', env=vec_env, learning_rate=float(ppo_cfg.get('learning_rate', 0.0003)), n_steps=int(ppo_cfg.get('n_steps', 256)), batch_size=int(ppo_cfg.get('batch_size', 64)), n_epochs=int(ppo_cfg.get('n_epochs', 10)), gamma=gamma, gae_lambda=float(ppo_cfg.get('gae_lambda', 0.95)), clip_range=float(ppo_cfg.get('clip_range', 0.2)), ent_coef=float(ppo_cfg.get('ent_coef', 0.01)), vf_coef=float(ppo_cfg.get('vf_coef', 0.5)), max_grad_norm=float(ppo_cfg.get('max_grad_norm', 0.5)), use_sde=bool(ppo_cfg.get('use_sde', True)), sde_sample_freq=int(ppo_cfg.get('sde_sample_freq', 4)), policy_kwargs=policy_kwargs, device=device, verbose=int(ppo_cfg.get('verbose', 0)))
