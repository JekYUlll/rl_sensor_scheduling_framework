from __future__ import annotations
from pathlib import Path
from typing import cast
import math
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from core.config import load_yaml, save_yaml
from core.seed import set_seed
from business_cases.windblown_case.predictor_targets import default_forecast_target_columns, default_reward_target_columns
from envs.truth_replay_env import TruthReplayConfig, TruthReplayEnvironment
from estimators.kalman_filter import KalmanFilterEstimator
from estimators.linear_gaussian_fit import fit_linear_gaussian_dynamics, safe_feature_scale, target_relevance_weights
from estimators.state_summary import flatten_rl_state
from evaluation.constraint_metrics import summarize_constraint_metrics
from reward.forecast_reward import FrozenForecastRewardEnsemble, FrozenForecastRewardOracle, load_reward_oracle, train_reward_oracle_suite_from_rollouts
from reward.mainline_reward import compute_forecast_task_terms, load_training_reward_cfg
from scheduling.action_space import DiscreteActionSpace
from scheduling.baselines.full_open_scheduler import FullOpenScheduler
from scheduling.baselines.info_priority_scheduler import InfoPriorityScheduler
from scheduling.online_projector import OnlineSubsetProjector
from scheduling.baselines.periodic_scheduler import PeriodicScheduler
from scheduling.baselines.random_scheduler import RandomScheduler
from scheduling.baselines.round_robin_scheduler import RoundRobinScheduler
from scheduling.rl.constrained_dqn_agent import ConstrainedDQNAgent
from scheduling.rl.dqn_agent import DQNAgent
from scheduling.rl.sb3_ppo import PPOPolicyAdapter, PPOTrainingCallback, WindblownSubsetGymEnv, build_ppo_model, make_vecnormalize, save_obs_normalization_stats
from scheduling.rl.score_dqn_agent import ConstrainedScoreDQNAgent, ScoreDQNAgent
from sensors.base_sensor import SensorSpec
from sensors.dataset_sensor import DatasetSensor
from visualization.estimation_plots import plot_trace_power
from visualization.policy_plots import plot_action_hist
from visualization.training_curves import plot_training_curves

def _build_run_dir(run_id: str) -> Path:
    root = Path('reports/runs') / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root
RewardOracleType = FrozenForecastRewardOracle | FrozenForecastRewardEnsemble
DEFAULT_BASE_CFG_PATH = 'configs/base.yaml'


class _ConstantSubsetScheduler:

    def __init__(self, subset: list[str]) -> None:
        self.subset = [str(sid) for sid in subset]

    def reset(self) -> None:
        return None

    def act(self, _state) -> list[str]:
        return list(self.subset)


class _RandomSubsetReplayScheduler:

    def __init__(
        self,
        selector: OnlineSubsetProjector,
        *,
        seed: int,
        hold_min: int,
        hold_max: int,
    ) -> None:
        self.selector = selector
        self.rng = np.random.default_rng(seed)
        self.hold_min = max(1, int(hold_min))
        self.hold_max = max(self.hold_min, int(hold_max))
        self.reset()

    def reset(self) -> None:
        self.current_subset: list[str] = []
        self.prev_selected: list[str] = []
        self.steps_until_resample = 0

    def _sample_subset(self) -> list[str]:
        feasible = self.selector.feasible_subsets(self.prev_selected, allow_empty=False)
        if not feasible:
            feasible = self.selector.feasible_subsets(None, allow_empty=False)
        if not feasible:
            return []
        idx = int(self.rng.integers(0, len(feasible)))
        return list(feasible[idx])

    def act(self, _state) -> list[str]:
        if self.steps_until_resample <= 0 or not self.current_subset:
            self.current_subset = self._sample_subset()
            self.steps_until_resample = int(self.rng.integers(self.hold_min, self.hold_max + 1))
        chosen = list(self.current_subset)
        self.prev_selected = list(chosen)
        self.steps_until_resample -= 1
        return chosen

def _split_bounds(n_rows: int, split_cfg: dict) -> dict[str, tuple[int, int]]:
    predictor_pretrain_ratio = float(split_cfg['predictor_pretrain_ratio'])
    rl_train_ratio = float(split_cfg['rl_train_ratio'])
    rl_val_ratio = float(split_cfg['rl_val_ratio'])
    final_test_ratio = float(split_cfg['final_test_ratio'])
    total_ratio = predictor_pretrain_ratio + rl_train_ratio + rl_val_ratio + final_test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f'Split ratios must sum to 1.0, got {total_ratio:.6f}')
    if min(predictor_pretrain_ratio, rl_train_ratio, rl_val_ratio, final_test_ratio) <= 0.0:
        raise ValueError('All split ratios must be positive')
    predictor_end = int(round(n_rows * predictor_pretrain_ratio))
    rl_train_end = predictor_end + int(round(n_rows * rl_train_ratio))
    rl_val_end = rl_train_end + int(round(n_rows * rl_val_ratio))
    rl_val_end = min(rl_val_end, n_rows)
    bounds = {
        'predictor_pretrain': (0, predictor_end),
        'rl_train': (predictor_end, rl_train_end),
        'rl_val': (rl_train_end, rl_val_end),
        'final_test': (rl_val_end, n_rows),
        'task_all': (predictor_end, n_rows),
        'all': (0, n_rows),
    }
    for split_name in ('predictor_pretrain', 'rl_train', 'rl_val', 'final_test'):
        start, end = bounds[split_name]
        if end <= start:
            raise ValueError(f'Split {split_name} is empty: {(start, end)}')
    return bounds

def _infer_state_columns(df: pd.DataFrame, env_cfg: dict) -> list[str]:
    configured = env_cfg.get('state_columns')
    if configured:
        return [str(col) for col in configured]
    exclude = {'t', 'time_idx', 'timestamp', 'storm_flag', 'event_flag'}
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(str(col))
    return numeric_cols

def _ensure_event_column(df: pd.DataFrame, env_cfg: dict) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    event_col = str(env_cfg.get('event_column', 'event_flag'))
    if event_col in out.columns:
        out[event_col] = out[event_col].astype(bool)
        return (out, event_col)
    if 'storm_flag' in out.columns:
        out[event_col] = out['storm_flag'].astype(bool)
        return (out, event_col)
    source_col = str(env_cfg.get('event_source_column', 'snow_mass_flux_kg_m2_s'))
    quantile = float(env_cfg.get('event_quantile', 0.9))
    threshold = float(out[source_col].quantile(quantile))
    out[event_col] = out[source_col] >= threshold
    return (out, event_col)

def _build_sensors(sensor_cfg: dict, state_columns: list[str]) -> list[DatasetSensor]:
    sensors: list[DatasetSensor] = []
    for item in sensor_cfg.get('sensors', []):
        variables = [str(v) for v in item.get('variables', [])]
        noise_std = item.get('noise_std', 0.0)
        if isinstance(noise_std, dict):
            obs_dim = len(variables)
        elif isinstance(noise_std, list):
            obs_dim = len(noise_std)
        else:
            obs_dim = len(variables)
        if obs_dim != len(variables):
            raise ValueError(f"Sensor {item['sensor_id']} has {len(variables)} variables but observation dimension {obs_dim}")
        spec = SensorSpec(sensor_id=str(item['sensor_id']), obs_dim=obs_dim, variables=variables, refresh_interval=int(item.get('refresh_interval', 1)), power_cost=float(item.get('power_cost', 1.0)), startup_delay=int(item.get('startup_delay', 0)), startup_peak_power=None if item.get('startup_peak_power') is None else float(item.get('startup_peak_power')), required=bool(item.get('required', False)), noise_std=noise_std)
        sensors.append(DatasetSensor(spec=spec, state_columns=state_columns))
    return sensors

def _sensor_to_dims(sensor_cfg: dict, state_columns: list[str]) -> dict[str, list[int]]:
    state_to_idx = {name: i for i, name in enumerate(state_columns)}
    out: dict[str, list[int]] = {}
    for sensor in sensor_cfg.get('sensors', []):
        dims = [state_to_idx[v] for v in sensor.get('variables', []) if v in state_to_idx]
        out[str(sensor['sensor_id'])] = dims
    return out

def _selector_mode(env_cfg: dict) -> str:
    return str(env_cfg.get('selector_mode', 'discrete')).strip().lower()

def _resolve_reward_target_columns(base_cfg: dict, env_cfg: dict, state_columns: list[str]) -> list[str]:
    configured = [str(col) for col in env_cfg.get('reward_target_columns', [])]
    if configured:
        return [col for col in configured if col in state_columns]
    if str(env_cfg.get('env_name', '')).strip().lower() == 'windblown':
        return [col for col in default_reward_target_columns() if col in state_columns]
    fallback = str(env_cfg.get('event_source_column', ''))
    if fallback and fallback in state_columns:
        return [fallback]
    return []

def _resolve_forecast_target_columns(env_cfg: dict, state_columns: list[str]) -> list[str]:
    configured = [str(col) for col in env_cfg.get('forecast_target_columns', [])]
    if configured:
        return configured
    if str(env_cfg.get('env_name', '')).strip().lower() == 'windblown':
        return default_forecast_target_columns()
    return list(state_columns)

def _make_selector(sensor_cfg: dict, base_cfg: dict, env_cfg: dict):
    sensor_ids = [str(s['sensor_id']) for s in sensor_cfg.get('sensors', [])]
    power_costs = {str(s['sensor_id']): float(s.get('power_cost', 1.0)) for s in sensor_cfg.get('sensors', [])}
    startup_peak_costs = {str(s['sensor_id']): float(s.get('startup_peak_power', s.get('power_cost', 1.0))) for s in sensor_cfg.get('sensors', [])}
    constraints = base_cfg.get('constraints', {})
    max_active = int(constraints.get('max_active', len(sensor_ids)))
    per_step_budget = float(constraints.get('per_step_budget', 2.0))
    startup_peak_budget = None if constraints.get('startup_peak_budget') is None else float(constraints.get('startup_peak_budget'))
    safety_margin = float(constraints.get('power_safety_margin', 0.0))
    selector_mode = _selector_mode(env_cfg)
    if selector_mode == 'online_projector':
        return OnlineSubsetProjector(sensor_ids=sensor_ids, power_costs=power_costs, max_active=max_active, per_step_budget=per_step_budget, startup_peak_costs=startup_peak_costs, startup_peak_budget=startup_peak_budget, safety_margin=safety_margin)
    required_sensor_ids = [str(s['sensor_id']) for s in sensor_cfg.get('sensors', []) if bool(s.get('required', False))]
    return DiscreteActionSpace(sensor_ids=sensor_ids, power_costs=power_costs, max_active=max_active, per_step_budget=per_step_budget, startup_peak_costs=startup_peak_costs, startup_peak_budget=startup_peak_budget, safety_margin=safety_margin, required_sensor_ids=required_sensor_ids)

def _build_estimator(truth_df: pd.DataFrame, state_columns: list[str], estimator_cfg: dict, sensor_cfg: dict, relevance_cfg: dict, reward_target_columns: list[str], x0: np.ndarray | None=None) -> KalmanFilterEstimator:
    values = truth_df[state_columns].to_numpy(dtype=float)
    a_mat, b_vec, q_mat = fit_linear_gaussian_dynamics(values, ridge_lambda=float(estimator_cfg.get('ridge_lambda', 0.0001)), fit_intercept=bool(estimator_cfg.get('fit_intercept', True)), max_spectral_radius=float(estimator_cfg.get('max_spectral_radius', 0.995)))
    p0_diag = estimator_cfg.get('P0_diag', [1.0])
    if len(p0_diag) == 1:
        p0 = np.diag(np.full(len(state_columns), float(p0_diag[0]), dtype=float))
    elif len(p0_diag) != len(state_columns):
        p0 = np.diag(np.full(len(state_columns), float(p0_diag[0]), dtype=float))
    else:
        p0 = np.diag(np.asarray(p0_diag, dtype=float))
    sensor_ids = [str(s['sensor_id']) for s in sensor_cfg.get('sensors', [])]
    state_scale = safe_feature_scale(values, min_scale=float(estimator_cfg.get('min_scale', 1e-06)))
    uncertainty_weights = target_relevance_weights(values, state_columns=state_columns, target_columns=reward_target_columns, min_weight=float(relevance_cfg.get('min_relevance_weight', 0.25)), power=float(relevance_cfg.get('relevance_power', 1.0)))
    return KalmanFilterEstimator(A=a_mat, Q=q_mat, x0=values[0] if x0 is None else np.asarray(x0, dtype=float), P0=p0, sensor_ids=sensor_ids, b=b_vec, state_scale=state_scale, uncertainty_weights=uncertainty_weights, normalize_rl_state=bool(estimator_cfg.get('normalize_rl_state', True)), use_logdet=bool(estimator_cfg.get('use_logdet', False)))

def _make_scheduler(scheduler_cfg: dict, selector, sensor_cfg: dict, state_columns: list[str]):
    name = str(scheduler_cfg.get('scheduler_name', 'random'))
    sensor_ids = [str(s['sensor_id']) for s in sensor_cfg.get('sensors', [])]
    max_active = int(getattr(selector, 'max_active', len(sensor_ids)))
    if name == 'random':
        return (RandomScheduler(selector), name)
    if name == 'periodic':
        return (PeriodicScheduler(selector, period=int(scheduler_cfg.get('period', 1))), name)
    if name == 'round_robin':
        return (RoundRobinScheduler(selector, sensor_ids=sensor_ids, max_active=max_active, min_on_steps=int(scheduler_cfg.get('min_on_steps', 1))), name)
    if name == 'info_priority':
        return (InfoPriorityScheduler(selector, sensor_ids=sensor_ids, sensor_to_dims=_sensor_to_dims(sensor_cfg, state_columns), max_active=max_active, weights=scheduler_cfg.get('weights', {})), name)
    if name == 'full_open':
        return (FullOpenScheduler(sensor_ids), name)
    if name in {'dqn', 'cmdp_dqn', 'ppo'}:
        return (None, name)
    raise ValueError(f'Unknown scheduler_name: {name}')

def _is_rl_scheduler(name: str) -> bool:
    return name in {'dqn', 'cmdp_dqn', 'ppo'}

def _build_rl_agent(name: str, state_dim: int, scheduler_cfg: dict, selector, episode_len: int | None=None):
    if isinstance(selector, OnlineSubsetProjector):
        if name == 'dqn':
            return ScoreDQNAgent(state_dim=state_dim, sensor_ids=list(selector.sensor_ids), cfg=scheduler_cfg, projector=selector)
        if name == 'cmdp_dqn':
            return ConstrainedScoreDQNAgent(state_dim=state_dim, sensor_ids=list(selector.sensor_ids), cfg=scheduler_cfg, projector=selector, device=str(scheduler_cfg.get('device', 'auto')), episode_len=episode_len)
        if name == 'ppo':
            return PPOPolicyAdapter(sensor_ids=list(selector.sensor_ids), cfg=scheduler_cfg, selector=selector)
    action_dim = selector.size()
    if name == 'dqn':
        return DQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=scheduler_cfg)
    if name == 'cmdp_dqn':
        return ConstrainedDQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=scheduler_cfg, steady_power_limit=selector.per_step_budget, episode_len=episode_len)
    raise ValueError(f'Unsupported RL scheduler: {name}')

def _optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {'', 'none', 'null'}:
        return None
    return float(value)

def _resolve_constraint_budgets(base_constraints: dict, scheduler_cfg: dict | None=None) -> dict[str, float | None]:
    scheduler_cfg = scheduler_cfg or {}
    cmdp_cfg = scheduler_cfg.get('cmdp', {}) if isinstance(scheduler_cfg, dict) else {}
    return {'peak_power_budget': _optional_float(base_constraints.get('startup_peak_budget')), 'average_power_budget': _optional_float(cmdp_cfg.get('average_power_budget', base_constraints.get('average_power_budget'))), 'episode_energy_budget': _optional_float(cmdp_cfg.get('episode_energy_budget', base_constraints.get('episode_energy_budget')))}

def _estimator_relevance_cfg(base_cfg: dict) -> dict[str, float]:
    reward_cfg = dict(base_cfg.get('reward', {}))
    return {
        'min_relevance_weight': float(reward_cfg.get('min_relevance_weight', 0.25)),
        'relevance_power': float(reward_cfg.get('relevance_power', 1.0)),
    }


def _task_reward_cfg(base_cfg: dict) -> dict[str, float]:
    return load_training_reward_cfg(base_cfg)

def _checkpoint_name(name: str) -> str:
    if name == 'ppo':
        return f'scheduler_{name}.zip'
    return f'scheduler_{name}.pt'

def _is_score_agent(agent) -> bool:
    return isinstance(agent, (ScoreDQNAgent, ConstrainedScoreDQNAgent, PPOPolicyAdapter))

def _is_constrained_agent(agent) -> bool:
    return isinstance(agent, (ConstrainedDQNAgent, ConstrainedScoreDQNAgent))

def _agent_act(agent, state_vec: np.ndarray, selector, prev_selected: list[str], greedy: bool):
    if _is_score_agent(agent):
        return agent.act(state_vec, greedy=greedy, prev_selected=prev_selected)
    feasible_ids = selector.feasible_action_ids(prev_selected)
    return agent.act(state_vec, greedy=greedy, feasible_action_ids=feasible_ids)

def _prev_selected_from_state(state: dict, selector) -> list[str]:
    prev_mask = state.get('previous_action', [])
    selected: list[str] = []
    for sid, flag in zip(getattr(selector, 'sensor_ids', []), prev_mask):
        if float(flag) > 0.5:
            selected.append(sid)
    return selected

def _resolve_action(action, selector, prev_selected: list[str] | None=None, scheduler_name: str | None=None) -> tuple[list[str], int | None]:
    if isinstance(selector, OnlineSubsetProjector):
        if isinstance(action, (list, tuple, np.ndarray)):
            subset = [str(sid) for sid in action]
            if scheduler_name == 'full_open':
                return (subset, None)
            if selector.is_subset_feasible(subset, prev_selected=prev_selected):
                return (subset, None)
            return (selector.project_ranked(subset, prev_selected=prev_selected), None)
        raise TypeError(f'Online projector expects subset/ranking action, got {type(action)!r}')
    if isinstance(action, (list, tuple)):
        ranked = [str(sid) for sid in action]
        if scheduler_name == 'full_open':
            return (ranked, None)
        aid = selector.nearest_feasible(ranked, prev_selected=prev_selected)
        return (selector.decode(aid), aid)
    action_id = selector.sanitize_action_id(int(action), prev_selected=prev_selected)
    return (selector.decode(action_id), action_id)

def _switch_count(prev_selected: list[str], selected: list[str]) -> int:
    return len(set(prev_selected) ^ set(selected))

def _state_tracking_loss(estimator: KalmanFilterEstimator, latent_state: dict[str, float], state_columns: list[str], reward_target_indices: list[int]) -> float:
    if not reward_target_indices:
        return 0.0
    truth = np.asarray([float(latent_state[col]) for col in state_columns], dtype=float)
    return estimator.normalized_state_error(truth, dims=reward_target_indices)

def _temporal_context(absolute_time_index: int, *, base_freq_s: int) -> dict[str, float]:
    seconds = float(absolute_time_index) * float(base_freq_s)
    phase = (seconds % 86400.0) / 86400.0
    theta = 2.0 * math.pi * phase
    return {
        'time_of_day_sin': float(math.sin(theta)),
        'time_of_day_cos': float(math.cos(theta)),
    }

def _current_rl_state(env, estimator: KalmanFilterEstimator, *, current_event: bool) -> dict:
    state = estimator.get_rl_state_features()
    state['event'] = bool(current_event)
    absolute_time_index = env.get_absolute_time_index() if hasattr(env, 'get_absolute_time_index') else env.get_time_index()
    base_freq_s = int(getattr(getattr(env, 'cfg', None), 'base_freq_s', 1))
    state.update(_temporal_context(int(absolute_time_index), base_freq_s=base_freq_s))
    return state

def _load_active_reward_oracle(base_cfg: dict, reward_artifact: str | None) -> RewardOracleType | None:
    reward_cfg = dict(base_cfg.get('forecast_reward', {}))
    enabled = bool(reward_cfg.get('enabled', False))
    if not enabled:
        raise ValueError('forecast_reward.enabled must be true for the current mainline')
    if not reward_artifact:
        raise ValueError('forecast_reward is enabled, but reward_artifact was not provided')
    artifact_path = Path(reward_artifact)
    if not artifact_path.exists():
        raise FileNotFoundError(f'Reward oracle artifact not found: {artifact_path}')
    return load_reward_oracle(artifact_path)

def _forecast_reward_loss(
    env: TruthReplayEnvironment,
    oracle: RewardOracleType | None,
    estimate_history: list[np.ndarray],
    observed_mask_history: list[np.ndarray] | None = None,
    time_index_history: list[int] | None = None,
    trace_history: list[float] | None = None,
    power_history: list[float] | None = None,
    peak_power_history: list[float] | None = None,
    event_history: list[int] | None = None,
) -> float:
    if oracle is None:
        return 0.0
    future_truth = env.peek_future_targets(oracle.horizon, list(env.state_columns))
    if not oracle.ready(len(estimate_history), len(future_truth)):
        return 0.0
    history = np.asarray(estimate_history[-oracle.lookback:], dtype=float)
    mask_history = None
    if observed_mask_history:
        mask_history = np.asarray(observed_mask_history[-oracle.lookback:], dtype=float)
    time_history = None
    if time_index_history:
        time_history = np.asarray(time_index_history[-oracle.lookback:], dtype=int)
    context_history: dict[str, np.ndarray] = {}
    if trace_history:
        context_history['trace_p'] = np.asarray(trace_history[-oracle.lookback:], dtype=float)
    if power_history:
        context_history['power'] = np.asarray(power_history[-oracle.lookback:], dtype=float)
    if peak_power_history:
        context_history['peak_power'] = np.asarray(peak_power_history[-oracle.lookback:], dtype=float)
    if event_history:
        context_history['event_flags'] = np.asarray(event_history[-oracle.lookback:], dtype=float)
    return oracle.score(
        history,
        future_truth,
        mask_history,
        time_history,
        context_history or None,
    )

def _build_truth_stack(truth_csv: str, env_cfg_path: str, sensor_cfg_path: str, estimator_cfg_path: str, split_name: str, seed: int, random_reset: bool, episode_len: int | None=None, base_cfg_path: str=DEFAULT_BASE_CFG_PATH):
    set_seed(seed)
    env_cfg = load_yaml(env_cfg_path)
    sensor_cfg = load_yaml(sensor_cfg_path)
    estimator_cfg = load_yaml(estimator_cfg_path)
    base_cfg = load_yaml(base_cfg_path)
    truth_df = pd.read_csv(truth_csv)
    truth_df, event_col = _ensure_event_column(truth_df, env_cfg)
    state_columns = _infer_state_columns(truth_df, env_cfg)
    relevance_cfg = _estimator_relevance_cfg(base_cfg)
    reward_cfg = _task_reward_cfg(base_cfg)
    reward_target_columns = _resolve_reward_target_columns(base_cfg, env_cfg, state_columns)
    split_cfg = base_cfg.get('data', {})
    bounds = _split_bounds(n_rows=len(truth_df), split_cfg=split_cfg)
    split_start, split_end = bounds[split_name]
    estimator_fit_split = str(split_cfg.get('estimator_fit_split', 'predictor_pretrain'))
    if estimator_fit_split not in bounds:
        raise ValueError(f'Unknown estimator_fit_split: {estimator_fit_split}')
    fit_start, fit_end = bounds[estimator_fit_split]
    sensors = _build_sensors(sensor_cfg, state_columns)
    run_cfg = base_cfg.get('run', {})
    env_episode_len = episode_len or min(int(run_cfg.get('episode_len', 400)), max(split_end - split_start, 1))
    env = TruthReplayEnvironment(truth_df=truth_df, sensors=sensors, cfg=TruthReplayConfig(state_columns=state_columns, split_start=split_start, split_end=split_end, episode_len=env_episode_len, random_reset=random_reset, base_freq_s=int(env_cfg.get('base_freq_s', 1)), event_column=event_col), seed=seed)
    estimator_train_df = truth_df.iloc[fit_start:fit_end].reset_index(drop=True)
    estimator = _build_estimator(estimator_train_df, state_columns=state_columns, estimator_cfg=estimator_cfg, sensor_cfg=sensor_cfg, relevance_cfg=relevance_cfg, reward_target_columns=reward_target_columns, x0=truth_df.iloc[split_start][state_columns].to_numpy(dtype=float))
    selector = _make_selector(sensor_cfg, base_cfg, env_cfg)
    col_to_idx = {name: i for i, name in enumerate(state_columns)}
    reward_target_indices = [col_to_idx[col] for col in reward_target_columns if col in col_to_idx]
    forecast_target_columns = _resolve_forecast_target_columns(env_cfg, state_columns)
    meta = {'truth_df': truth_df, 'state_columns': state_columns, 'event_column': event_col, 'bounds': bounds, 'base_cfg': base_cfg, 'env_cfg': env_cfg, 'sensor_cfg': sensor_cfg, 'reward_cfg': reward_cfg, 'reward_target_columns': reward_target_columns, 'reward_target_indices': reward_target_indices, 'forecast_target_columns': forecast_target_columns, 'estimator_fit_split': estimator_fit_split}
    return (env, estimator, selector, meta)

def _rollout_scheduler(env, estimator, scheduler, selector, reward_cfg: dict, reward_target_indices: list[int], reward_oracle: RewardOracleType | None=None, greedy: bool=False, collect_series: bool=False, scheduler_name: str | None=None):
    reset_out = env.reset()
    estimator.reset()
    scheduler.reset()
    current_event = bool(reset_out.get('event_flags', {}).get('event', False))
    state_columns = getattr(env, 'state_columns', [])
    col_to_idx = {name: i for i, name in enumerate(state_columns)}
    estimate_history: list[np.ndarray] = [estimator.get_state_estimate().copy()]
    observed_mask_history: list[np.ndarray] = [np.ones(len(state_columns), dtype=float)]
    time_index_history: list[int] = [int(env.get_absolute_time_index()) if hasattr(env, 'get_absolute_time_index') else int(env.get_time_index())]
    initial_trace = float(estimator.get_uncertainty_summary()['trace_P'])
    trace_context_history: list[float] = [initial_trace]
    power_context_history: list[float] = [0.0]
    peak_power_context_history: list[float] = [0.0]
    event_context_history: list[int] = [1 if current_event else 0]
    total_reward = 0.0
    action_ids: list[int] = []
    trace_hist: list[float] = []
    uncertainty_hist: list[float] = []
    forecast_hist: list[float] = []
    task_loss_hist: list[float] = []
    switch_penalty_hist: list[float] = []
    coverage_penalty_hist: list[float] = []
    violation_penalty_hist: list[float] = []
    state_tracking_hist: list[float] = []
    power_hist: list[float] = []
    peak_power_hist: list[float] = []
    startup_extra_hist: list[float] = []
    coverage_hist: list[float] = []
    truth_hist: list[list[float]] = []
    estimate_hist: list[list[float]] = []
    observed_mask_hist: list[list[float]] = []
    event_hist: list[int] = []
    time_index_hist: list[int] = []
    prev_selected: list[str] = []
    while True:
        state = _current_rl_state(env, estimator, current_event=current_event)
        raw_action = scheduler.act({**state, 't': env.get_time_index()}) if greedy else scheduler.act(state)
        selected, action_id = _resolve_action(raw_action, selector, prev_selected=prev_selected, scheduler_name=scheduler_name)
        if action_id is not None:
            action_ids.append(action_id)
        power_info = selector.power_metrics(selected, prev_selected=prev_selected)
        step = env.step(selected)
        estimator.predict()
        estimator.update(step['available_observations'])
        observed_sensor_ids = [str(obs['sensor_id']) for obs in step['available_observations'] if obs.get('available', False)]
        power_cost = float(power_info['steady_power'])
        power_ratio = power_cost / max(float(getattr(selector, 'per_step_budget', power_cost or 1.0)), 1e-06)
        estimator.on_step(selected_sensor_ids=selected, power_ratio=power_ratio, observed_sensor_ids=observed_sensor_ids)
        current_event = bool(step.get('event_flags', {}).get('event', False))
        next_state = _current_rl_state(env, estimator, current_event=current_event)
        unc_summary = estimator.get_uncertainty_summary()
        trace_p = float(unc_summary['trace_P'])
        estimate_history.append(estimator.get_state_estimate().copy())
        state_tracking_loss = 0.0
        if float(reward_cfg.get('lambda_state_tracking', 0.0)) > 0.0:
            state_tracking_loss = _state_tracking_loss(estimator, latent_state=step['latent_state'], state_columns=state_columns, reward_target_indices=reward_target_indices)
        mask = np.zeros(len(state_columns), dtype=float)
        for obs in step['available_observations']:
            for var_name in obs.get('variables', []):
                idx = col_to_idx.get(var_name)
                if idx is not None:
                    mask[idx] = 1.0
        observed_mask_history.append(mask.copy())
        time_index_history.append(int(env.get_absolute_time_index()) if hasattr(env, 'get_absolute_time_index') else int(env.get_time_index()))
        trace_context_history.append(trace_p)
        power_context_history.append(power_cost)
        peak_power_context_history.append(float(power_info['peak_power']))
        event_context_history.append(1 if current_event else 0)
        forecast_loss = _forecast_reward_loss(env, reward_oracle, estimate_history, observed_mask_history, time_index_history, trace_context_history, power_context_history, peak_power_context_history, event_context_history)
        reward_terms = compute_forecast_task_terms(
            forecast_loss=forecast_loss,
            switch_count=_switch_count(prev_selected, selected),
            coverage_ratio=next_state.get('coverage_ratio', []),
            steady_power=power_cost,
            peak_power=float(power_info['peak_power']),
            steady_limit=max(float(getattr(selector, 'per_step_budget', power_cost or 1.0)) - float(getattr(selector, 'safety_margin', 0.0)), 0.0),
            peak_limit=None if getattr(selector, 'startup_peak_budget', None) is None else max(float(selector.startup_peak_budget) - float(getattr(selector, 'safety_margin', 0.0)), 0.0),
            reward_cfg=reward_cfg,
            state_tracking_loss=state_tracking_loss,
        )
        reward = float(reward_terms['task_reward'])
        total_reward += reward
        trace_hist.append(trace_p)
        uncertainty_hist.append(float(unc_summary.get('weighted_trace_P_norm', trace_p)))
        forecast_hist.append(float(forecast_loss))
        task_loss_hist.append(float(reward_terms['task_loss']))
        switch_penalty_hist.append(float(reward_terms['switch_penalty_raw']))
        coverage_penalty_hist.append(float(reward_terms['coverage_penalty_raw']))
        violation_penalty_hist.append(float(reward_terms['violation_penalty_raw']))
        state_tracking_hist.append(float(reward_terms['state_tracking_loss']))
        power_hist.append(power_cost)
        peak_power_hist.append(float(power_info['peak_power']))
        startup_extra_hist.append(float(power_info['startup_extra_power']))
        cov = next_state.get('coverage_ratio', [])
        coverage_hist.append(float(np.mean(cov)) if cov else 0.0)
        if collect_series:
            truth_hist.append([float(step['latent_state'][col]) for col in state_columns])
            estimate_hist.append([float(v) for v in estimator.get_state_estimate().tolist()])
            observed_mask_hist.append(mask.tolist())
            event_hist.append(1 if bool(step.get('event_flags', {}).get('event', False)) else 0)
            time_index_hist.append(time_index_history[-1])
        prev_selected = list(selected)
        if step['done']:
            break
    return {'episode_reward': total_reward, 'trace_hist': trace_hist, 'uncertainty_hist': uncertainty_hist, 'forecast_hist': forecast_hist, 'task_loss_hist': task_loss_hist, 'switch_penalty_hist': switch_penalty_hist, 'coverage_penalty_hist': coverage_penalty_hist, 'violation_penalty_hist': violation_penalty_hist, 'state_tracking_hist': state_tracking_hist, 'power_hist': power_hist, 'peak_power_hist': peak_power_hist, 'startup_extra_hist': startup_extra_hist, 'coverage_hist': coverage_hist, 'action_ids': action_ids, 'truth_hist': truth_hist, 'estimate_hist': estimate_hist, 'observed_mask_hist': observed_mask_hist, 'event_hist': event_hist, 'time_index_hist': time_index_hist}

def _make_greedy_agent_scheduler(agent, selector, scheduler_name: str):

    class _GreedyPolicy:

        def reset(self):
            return None

        def act(self, state):
            vec = np.asarray(flatten_rl_state(state), dtype=np.float32)
            prev_selected = _prev_selected_from_state(state, selector)
            return _agent_act(agent, vec, selector, prev_selected, greedy=True)
    return _GreedyPolicy()

def _validation_objective(summary: dict, constrained: bool) -> float:
    objective = float(summary.get('task_reward_mean', summary['reward_mean']))
    if constrained:
        objective -= 10000.0 * max(float(summary.get('avg_power_violation', 0.0)), 0.0)
        objective -= 10000.0 * max(float(summary.get('peak_violation_rate', 0.0)), 0.0)
    return objective

def _evaluate_agent_on_split(truth_csv: str, env_cfg_path: str, sensor_cfg_path: str, estimator_cfg_path: str, scheduler_cfg_path: str, split_name: str, agent, reward_artifact: str | None, base_cfg_path: str=DEFAULT_BASE_CFG_PATH) -> dict[str, float]:
    seed = base_cfg_seed(base_cfg_path)
    env, estimator, selector, meta = _build_truth_stack(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, split_name=split_name, seed=seed, random_reset=False, episode_len=meta_length_from_truth_csv(truth_csv, env_cfg_path, split_name=split_name, base_cfg_path=base_cfg_path), base_cfg_path=base_cfg_path)
    scheduler_cfg = load_yaml(scheduler_cfg_path)
    _, name = _make_scheduler(scheduler_cfg, selector, meta['sensor_cfg'], meta['state_columns'])
    base_cfg = meta['base_cfg']
    constraints_cfg = base_cfg.get('constraints', {})
    constraint_budgets = _resolve_constraint_budgets(constraints_cfg, scheduler_cfg)
    task_reward_cfg = _task_reward_cfg(base_cfg)
    reward_oracle = _load_active_reward_oracle(base_cfg, reward_artifact)
    scheduler = _make_greedy_agent_scheduler(agent, selector, name)
    out = _rollout_scheduler(env, estimator, scheduler, selector, task_reward_cfg, reward_target_indices=list(meta.get('reward_target_indices', [])), reward_oracle=reward_oracle, greedy=True, scheduler_name=name)
    constraint_metrics = summarize_constraint_metrics(steady_power_hist=out['power_hist'], peak_power_hist=out['peak_power_hist'], startup_extra_hist=out['startup_extra_hist'], average_power_budget=constraint_budgets['average_power_budget'], episode_energy_budget=constraint_budgets['episode_energy_budget'], peak_power_budget=constraint_budgets['peak_power_budget'])
    return {'reward_mean': float(out['episode_reward']), 'task_reward_mean': float(out['episode_reward']), 'task_loss_mean': float(np.mean(out['task_loss_hist'])) if out['task_loss_hist'] else float('nan'), 'trace_P_mean': float(np.mean(out['trace_hist'])) if out['trace_hist'] else float('nan'), 'uncertainty_mean': float(np.mean(out['uncertainty_hist'])) if out['uncertainty_hist'] else float('nan'), 'forecast_loss_mean': float(np.mean(out['forecast_hist'])) if out['forecast_hist'] else float('nan'), 'switch_penalty_mean': float(np.mean(out['switch_penalty_hist'])) if out['switch_penalty_hist'] else float('nan'), 'coverage_penalty_mean': float(np.mean(out['coverage_penalty_hist'])) if out['coverage_penalty_hist'] else float('nan'), 'constraint_violation_mean': float(np.mean(out['violation_penalty_hist'])) if out['violation_penalty_hist'] else float('nan'), 'state_tracking_loss_mean': float(np.mean(out['state_tracking_hist'])) if out['state_tracking_hist'] else float('nan'), 'power_mean': float(constraint_metrics['power_mean']), 'peak_power_max': float(constraint_metrics['peak_power_max']), 'total_energy': float(constraint_metrics['total_energy']), 'peak_violation_rate': float(constraint_metrics['peak_violation_rate']), 'avg_power_violation': float(constraint_metrics['avg_power_violation']), 'energy_violation': float(constraint_metrics['energy_violation']), 'coverage_mean': float(np.mean(out['coverage_hist'])) if out['coverage_hist'] else float('nan')}

def base_cfg_seed(base_cfg_path: str=DEFAULT_BASE_CFG_PATH) -> int:
    base = load_yaml(base_cfg_path)
    return int(base.get('seed', 42))


def _append_reward_pretrain_rollout(
    *,
    env,
    estimator,
    selector,
    scheduler,
    scheduler_name: str,
    reward_cfg: dict,
    reward_target_indices: list[int],
    reward_rollouts: list[dict[str, np.ndarray]],
    rollout_meta: list[dict[str, float | str]],
    rollout_idx: int,
) -> None:
    rollout = _rollout_scheduler(
        env,
        estimator,
        scheduler,
        selector,
        reward_cfg=reward_cfg,
        reward_target_indices=reward_target_indices,
        reward_oracle=None,
        greedy=True,
        collect_series=True,
        scheduler_name=scheduler_name,
    )
    if not rollout['estimate_hist'] or not rollout['truth_hist']:
        raise ValueError(
            f"Reward predictor pretraining rollout produced empty series for scheduler '{scheduler_name}'"
        )
    estimate_arr = np.asarray(rollout['estimate_hist'], dtype=float)
    truth_arr = np.asarray(rollout['truth_hist'], dtype=float)
    mask_arr = np.asarray(rollout['observed_mask_hist'], dtype=float)
    time_index_arr = np.asarray(rollout['time_index_hist'], dtype=int)
    reward_rollouts.append(
        {
            'input_series': estimate_arr,
            'target_series': truth_arr,
            'observed_mask': mask_arr,
            'trace_p': np.asarray(rollout['trace_hist'], dtype=float),
            'power': np.asarray(rollout['power_hist'], dtype=float),
            'peak_power': np.asarray(rollout['peak_power_hist'], dtype=float),
            'event_flags': np.asarray(rollout['event_hist'], dtype=int),
            'time_index': time_index_arr,
        }
    )
    rollout_meta.append(
        {
            'scheduler': scheduler_name,
            'rollout_idx': float(rollout_idx),
            'num_steps': float(estimate_arr.shape[0]),
            'coverage_mean': float(np.mean(rollout['coverage_hist'])) if rollout['coverage_hist'] else float('nan'),
            'power_mean': float(np.mean(rollout['power_hist'])) if rollout['power_hist'] else float('nan'),
        }
    )

def pretrain_reward_predictor(truth_csv: str, env_cfg_path: str, sensor_cfg_path: str, estimator_cfg_path: str, reward_cfg_path: str, run_id: str, base_cfg_path: str=DEFAULT_BASE_CFG_PATH) -> dict:
    seed = base_cfg_seed(base_cfg_path)
    reward_cfg = load_yaml(reward_cfg_path)
    pretrain_split = str(reward_cfg.get('pretrain_split', 'predictor_pretrain'))
    if pretrain_split != 'predictor_pretrain':
        raise ValueError(
            f"Reward pretraining must use split 'predictor_pretrain', got '{pretrain_split}'"
        )
    pretrain_episode_len_cfg = reward_cfg.get('pretrain_episode_len')
    pretrain_rollouts_per_scheduler = max(1, int(reward_cfg.get('pretrain_rollouts_per_scheduler', 1)))
    random_subset_rollouts = max(0, int(reward_cfg.get('pretrain_random_subset_rollouts', 0)))
    random_subset_hold_min = max(1, int(reward_cfg.get('pretrain_random_subset_hold_min', 1)))
    random_subset_hold_max = max(random_subset_hold_min, int(reward_cfg.get('pretrain_random_subset_hold_max', random_subset_hold_min)))
    all_feasible_subset_rollouts = bool(reward_cfg.get('pretrain_all_feasible_subsets', False))
    max_feasible_constant_rollouts = max(1, int(reward_cfg.get('pretrain_max_feasible_constant_rollouts', 64)))
    random_reset = bool(reward_cfg.get('pretrain_random_reset', pretrain_rollouts_per_scheduler > 1))
    split_len = meta_length_from_truth_csv(truth_csv, env_cfg_path, split_name='predictor_pretrain', base_cfg_path=base_cfg_path)
    pretrain_episode_len = split_len if pretrain_episode_len_cfg is None else min(int(pretrain_episode_len_cfg), split_len)
    env, estimator, selector, meta = _build_truth_stack(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, split_name='predictor_pretrain', seed=seed, random_reset=random_reset, episode_len=pretrain_episode_len, base_cfg_path=base_cfg_path)
    reward_cfg['base_freq_s'] = int(meta['env_cfg'].get('base_freq_s', 1))
    run_dir = _build_run_dir(run_id)
    scheduler_names = [str(name) for name in reward_cfg.get('pretrain_schedulers', ['full_open'])]
    reward_rollouts: list[dict[str, np.ndarray]] = []
    rollout_meta: list[dict[str, float | str]] = []
    for sched_idx, sched_name in enumerate(scheduler_names):
        scheduler_cfg = load_yaml(f'configs/scheduler/{sched_name}.yaml')
        scheduler, resolved_name = _make_scheduler(scheduler_cfg, selector, meta['sensor_cfg'], meta['state_columns'])
        if scheduler is None:
            raise ValueError(f"Reward pretraining does not support RL scheduler '{sched_name}'")
        for rollout_idx in range(pretrain_rollouts_per_scheduler):
            np.random.seed(seed + sched_idx * 100 + rollout_idx)
            _append_reward_pretrain_rollout(
                env=env,
                estimator=estimator,
                selector=selector,
                scheduler=scheduler,
                scheduler_name=resolved_name,
                reward_cfg=meta['reward_cfg'],
                reward_target_indices=list(meta.get('reward_target_indices', [])),
                reward_rollouts=reward_rollouts,
                rollout_meta=rollout_meta,
                rollout_idx=rollout_idx,
            )
    extra_rollout_idx = len(rollout_meta)
    if all_feasible_subset_rollouts:
        if not isinstance(selector, OnlineSubsetProjector):
            raise TypeError('pretrain_all_feasible_subsets requires the online subset projector path')
        feasible_subsets = [
            list(subset)
            for subset in selector.feasible_subsets(None, allow_empty=False)
        ]
        if len(feasible_subsets) > max_feasible_constant_rollouts:
            rng = np.random.default_rng(seed + 10_000)
            chosen_idx = rng.choice(len(feasible_subsets), size=max_feasible_constant_rollouts, replace=False)
            feasible_subsets = [feasible_subsets[int(idx)] for idx in np.sort(chosen_idx)]
        for subset in feasible_subsets:
            scheduler = _ConstantSubsetScheduler(subset)
            subset_name = 'subset_' + ('+'.join(subset) if subset else 'empty')
            _append_reward_pretrain_rollout(
                env=env,
                estimator=estimator,
                selector=selector,
                scheduler=scheduler,
                scheduler_name=subset_name,
                reward_cfg=meta['reward_cfg'],
                reward_target_indices=list(meta.get('reward_target_indices', [])),
                reward_rollouts=reward_rollouts,
                rollout_meta=rollout_meta,
                rollout_idx=extra_rollout_idx,
            )
            extra_rollout_idx += 1
    if random_subset_rollouts > 0:
        if not isinstance(selector, OnlineSubsetProjector):
            raise TypeError('pretrain_random_subset_rollouts requires the online subset projector path')
        for rollout_idx in range(random_subset_rollouts):
            scheduler = _RandomSubsetReplayScheduler(
                selector,
                seed=seed + 20_000 + rollout_idx,
                hold_min=random_subset_hold_min,
                hold_max=random_subset_hold_max,
            )
            _append_reward_pretrain_rollout(
                env=env,
                estimator=estimator,
                selector=selector,
                scheduler=scheduler,
                scheduler_name='subset_random',
                reward_cfg=meta['reward_cfg'],
                reward_target_indices=list(meta.get('reward_target_indices', [])),
                reward_rollouts=reward_rollouts,
                rollout_meta=rollout_meta,
                rollout_idx=extra_rollout_idx,
            )
            extra_rollout_idx += 1
    if not reward_rollouts:
        raise ValueError('Reward predictor pretraining produced no rollout data')
    state_columns = list(meta['state_columns'])
    target_columns = [str(col) for col in reward_cfg.get('target_columns', meta.get('reward_target_columns', []))]
    if not target_columns:
        target_columns = list(meta.get('reward_target_columns', []))
    reward_out = train_reward_oracle_suite_from_rollouts(rollouts=reward_rollouts, input_columns=state_columns, target_columns=target_columns, reward_cfg=reward_cfg, artifact_dir=run_dir)
    entry_metrics = []
    for entry in reward_out.get('entries', []):
        metrics = dict(entry.get('metrics', {}))
        entry_metrics.append(float(metrics.get('rmse', np.nan)))
    rmse_mean = float(np.nanmean(entry_metrics)) if entry_metrics else float('nan')
    summary = {'run_id': run_id, 'reward_cfg': reward_cfg_path, 'artifact_path': reward_out['artifact_path'], 'predictor_names': [str(entry.get('model_name', '')) for entry in reward_out.get('entries', [])], 'lookback': reward_out['lookback'], 'horizon': reward_out['horizon'], 'target_columns': reward_out['target_columns'], 'pretrain_schedulers': scheduler_names, 'pretrain_rollouts_per_scheduler': int(pretrain_rollouts_per_scheduler), 'pretrain_random_subset_rollouts': int(random_subset_rollouts), 'pretrain_all_feasible_subsets': bool(all_feasible_subset_rollouts), 'pretrain_episode_len': int(pretrain_episode_len), 'num_rollout_steps': int(sum((int(item['input_series'].shape[0]) for item in reward_rollouts))), 'oracle_count': int(len(reward_out.get('entries', []))), 'rmse_mean': rmse_mean}
    save_yaml(summary, run_dir / 'reward_predictor_meta.yaml')
    pd.DataFrame([summary]).to_csv(run_dir / 'reward_predictor_metrics.csv', index=False)
    entry_rows: list[dict[str, float | str]] = []
    for entry in reward_out.get('entries', []):
        row: dict[str, float | str] = {'model_name': str(entry.get('model_name', '')), 'weight': float(entry.get('weight', 0.0)), 'artifact_path': str(entry.get('artifact_path', ''))}
        for key, value in dict(entry.get('metrics', {})).items():
            row[str(key)] = float(value)
        entry_rows.append(row)
    if entry_rows:
        pd.DataFrame(entry_rows).to_csv(run_dir / 'reward_predictor_model_metrics.csv', index=False)
    pd.DataFrame(rollout_meta).to_csv(run_dir / 'reward_pretrain_rollouts.csv', index=False)
    return {'run_dir': str(run_dir), 'summary': summary}

def _train_ppo_scheduler(*, truth_csv: str, env_cfg_path: str, sensor_cfg_path: str, estimator_cfg_path: str, scheduler_cfg_path: str, run_dir: Path, selector, scheduler_cfg: dict, task_reward_cfg: dict, reward_target_indices: list[int], reward_artifact: str | None, base_seed: int, episode_len: int, base_cfg_path: str=DEFAULT_BASE_CFG_PATH) -> dict:
    if not isinstance(selector, OnlineSubsetProjector):
        raise TypeError('PPO baseline currently supports only the windblown online-subset projector path')
    projector = selector
    base_cfg = load_yaml(base_cfg_path)
    reward_oracle = _load_active_reward_oracle(base_cfg, reward_artifact)
    constraint_budgets = _resolve_constraint_budgets(base_cfg.get('constraints', {}), scheduler_cfg)

    def _make_train_env(seed_offset: int) -> WindblownSubsetGymEnv:
        env_train, estimator_train, _, meta_train = _build_truth_stack(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, split_name='rl_train', seed=base_seed + seed_offset, random_reset=True, base_cfg_path=base_cfg_path)
        return WindblownSubsetGymEnv(env=env_train, estimator=estimator_train, selector=projector, task_reward_cfg=task_reward_cfg, state_columns=list(meta_train['state_columns']), reward_target_indices=list(reward_target_indices), constraint_budgets=constraint_budgets, reward_oracle=reward_oracle)
    train_template_env = _make_train_env(0)
    eval_interval = int(scheduler_cfg.get('ppo', {}).get('eval_interval_episodes', 5))
    eval_episodes = int(scheduler_cfg.get('ppo', {}).get('eval_episodes', 3))
    total_timesteps = int(scheduler_cfg.get('ppo', {}).get('total_timesteps', int(base_cfg.get('run', {}).get('num_episodes', 80)) * max(episode_len, 1)))

    def _eval_env_factory():
        env_val, estimator_val, _, meta_val = _build_truth_stack(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, split_name='rl_val', seed=base_seed, random_reset=False, episode_len=meta_length_from_truth_csv(truth_csv, env_cfg_path, split_name='rl_val', base_cfg_path=base_cfg_path), base_cfg_path=base_cfg_path)
        raw_env = WindblownSubsetGymEnv(env=env_val, estimator=estimator_val, selector=projector, task_reward_cfg=task_reward_cfg, state_columns=list(meta_val['state_columns']), reward_target_indices=list(meta_val.get('reward_target_indices', [])), constraint_budgets=constraint_budgets, reward_oracle=reward_oracle)
        ppo_cfg = dict(scheduler_cfg.get('ppo', {}))
        gamma = float(ppo_cfg.get('gamma', 0.99))
        return make_vecnormalize(DummyVecEnv([lambda env=raw_env: env]), gamma=gamma, norm_obs=bool(ppo_cfg.get('normalize_observations', True)), norm_reward=False, clip_obs=float(ppo_cfg.get('clip_obs', 10.0)), clip_reward=float(ppo_cfg.get('clip_reward', 10.0)), training=False)
    best_ckpt = run_dir / _checkpoint_name('ppo')
    last_ckpt = run_dir / 'scheduler_ppo_last.zip'
    norm_stats_path = best_ckpt.with_suffix('.norm.npz')
    ppo_cfg = dict(scheduler_cfg.get('ppo', {}))
    n_envs = max(1, int(ppo_cfg.get('n_envs', 4)))
    env_fns = [lambda seed_offset=i: _make_train_env(seed_offset) for i in range(n_envs)]
    model = build_ppo_model(scheduler_cfg, env_fns, device=str(scheduler_cfg.get('device', 'auto')))
    callback = PPOTrainingCallback(run_dir=run_dir, eval_env_factory=_eval_env_factory, eval_interval_episodes=eval_interval, eval_episodes=eval_episodes, best_model_path=best_ckpt)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    model.save(str(last_ckpt))
    train_vec_env = model.get_env()
    if train_vec_env is None:
        raise RuntimeError('PPO model did not retain its vectorized environment')
    train_norm = cast(VecNormalize, train_vec_env)
    save_obs_normalization_stats(train_norm, last_ckpt.with_suffix('.norm.npz'))
    callback.export_training_log()
    if not best_ckpt.exists():
        model.save(str(best_ckpt))
        save_obs_normalization_stats(train_norm, norm_stats_path)
    elif not norm_stats_path.exists():
        save_obs_normalization_stats(train_norm, norm_stats_path)
    adapter = PPOPolicyAdapter(sensor_ids=list(projector.sensor_ids), cfg=scheduler_cfg, selector=projector)
    adapter.load(str(best_ckpt))
    val_summary = _evaluate_agent_on_split(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, scheduler_cfg_path=scheduler_cfg_path, split_name='rl_val', agent=adapter, reward_artifact=reward_artifact, base_cfg_path=base_cfg_path)
    metrics = {'scheduler': 'ppo', 'reward_mean': float(val_summary['reward_mean']), 'task_reward_mean': float(val_summary['task_reward_mean']), 'task_loss_mean': float(val_summary['task_loss_mean']), 'trace_P_mean': float(val_summary['trace_P_mean']), 'uncertainty_mean': float(val_summary['uncertainty_mean']), 'forecast_loss_mean': float(val_summary['forecast_loss_mean']), 'switch_penalty_mean': float(val_summary['switch_penalty_mean']), 'coverage_penalty_mean': float(val_summary['coverage_penalty_mean']), 'constraint_violation_mean': float(val_summary['constraint_violation_mean']), 'state_tracking_loss_mean': float(val_summary['state_tracking_loss_mean']), 'power_mean': float(val_summary['power_mean']), 'peak_power_max_mean': float(val_summary['peak_power_max']), 'total_energy_mean': float(val_summary['total_energy']), 'peak_violation_rate_mean': float(val_summary['peak_violation_rate']), 'avg_power_violation_mean': float(val_summary['avg_power_violation']), 'energy_violation_mean': float(val_summary['energy_violation']), 'coverage_mean': float(val_summary['coverage_mean'])}
    pd.DataFrame([metrics]).to_csv(run_dir / 'metrics_estimation.csv', index=False)
    return {'metrics': metrics, 'checkpoint': str(best_ckpt)}

def run_scheduler_training(truth_csv: str, env_cfg_path: str, sensor_cfg_path: str, estimator_cfg_path: str, scheduler_cfg_path: str, run_id: str, reward_artifact: str | None=None, base_cfg_path: str=DEFAULT_BASE_CFG_PATH) -> dict:
    seed = base_cfg_seed(base_cfg_path)
    env, estimator, selector, meta = _build_truth_stack(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, split_name='rl_train', seed=seed, random_reset=True, base_cfg_path=base_cfg_path)
    base_cfg = meta['base_cfg']
    scheduler_cfg = load_yaml(scheduler_cfg_path)
    scheduler, name = _make_scheduler(scheduler_cfg, selector, meta['sensor_cfg'], meta['state_columns'])
    run_dir = _build_run_dir(run_id)
    save_yaml({'truth_csv': truth_csv, 'base_cfg': base_cfg_path, 'env_cfg': env_cfg_path, 'sensor_cfg': sensor_cfg_path, 'estimator_cfg': estimator_cfg_path, 'scheduler_cfg': scheduler_cfg_path, 'reward_artifact': reward_artifact}, run_dir / 'config_used.yaml')
    run_cfg = base_cfg.get('run', {})
    constraints_cfg = base_cfg.get('constraints', {})
    constraint_budgets = _resolve_constraint_budgets(constraints_cfg, scheduler_cfg)
    task_reward_cfg = _task_reward_cfg(base_cfg)
    reward_target_indices = list(meta.get('reward_target_indices', []))
    reward_oracle = _load_active_reward_oracle(base_cfg, reward_artifact)
    if name == 'ppo':
        ppo_out = _train_ppo_scheduler(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, scheduler_cfg_path=scheduler_cfg_path, run_dir=run_dir, selector=selector, scheduler_cfg=scheduler_cfg, task_reward_cfg=task_reward_cfg, reward_target_indices=reward_target_indices, reward_artifact=reward_artifact, base_seed=seed, episode_len=int(run_cfg.get('episode_len', 400)), base_cfg_path=base_cfg_path)
        metrics = ppo_out['metrics']
        return {'run_dir': str(run_dir), 'metrics': metrics, 'scheduler': name, 'checkpoint': ppo_out['checkpoint']}
    assert name != 'ppo'
    if not _is_rl_scheduler(name):
        episodes = int(run_cfg.get('eval_episodes', 10))
        rewards = []
        traces = []
        uncertainties = []
        forecasts = []
        powers = []
        peaks = []
        startup_extras = []
        coverages = []
        total_energies = []
        peak_violation_rates = []
        avg_power_violation_hist = []
        energy_violation_hist = []
        task_losses = []
        switch_penalties = []
        coverage_penalties = []
        violation_penalties = []
        state_tracking_losses = []
        all_actions = []
        for _ in range(episodes):
            out = _rollout_scheduler(env, estimator, scheduler, selector, task_reward_cfg, reward_target_indices=reward_target_indices, reward_oracle=reward_oracle, greedy=True, scheduler_name=name)
            rewards.append(out['episode_reward'])
            traces.append(float(np.mean(out['trace_hist'])))
            uncertainties.append(float(np.mean(out['uncertainty_hist'])))
            forecasts.append(float(np.mean(out['forecast_hist'])))
            task_losses.append(float(np.mean(out['task_loss_hist'])))
            switch_penalties.append(float(np.mean(out['switch_penalty_hist'])))
            coverage_penalties.append(float(np.mean(out['coverage_penalty_hist'])))
            violation_penalties.append(float(np.mean(out['violation_penalty_hist'])))
            state_tracking_losses.append(float(np.mean(out['state_tracking_hist'])))
            coverages.append(float(np.mean(out['coverage_hist'])))
            all_actions.extend(out['action_ids'])
            constraint_metrics = summarize_constraint_metrics(steady_power_hist=out['power_hist'], peak_power_hist=out['peak_power_hist'], startup_extra_hist=out['startup_extra_hist'], average_power_budget=constraint_budgets['average_power_budget'], episode_energy_budget=constraint_budgets['episode_energy_budget'], peak_power_budget=constraint_budgets['peak_power_budget'])
            powers.append(float(constraint_metrics['power_mean']))
            peaks.append(float(constraint_metrics['peak_power_max']))
            startup_extras.append(float(constraint_metrics['startup_extra_power_mean']))
            total_energies.append(float(constraint_metrics['total_energy']))
            peak_violation_rates.append(float(constraint_metrics['peak_violation_rate']))
            avg_power_violation_hist.append(float(constraint_metrics['avg_power_violation']))
            energy_violation_hist.append(float(constraint_metrics['energy_violation']))
        metrics = {'scheduler': name, 'reward_mean': float(np.mean(rewards)), 'task_reward_mean': float(np.mean(rewards)), 'task_loss_mean': float(np.mean(task_losses)), 'trace_P_mean': float(np.mean(traces)), 'uncertainty_mean': float(np.mean(uncertainties)), 'forecast_loss_mean': float(np.mean(forecasts)), 'switch_penalty_mean': float(np.mean(switch_penalties)), 'coverage_penalty_mean': float(np.mean(coverage_penalties)), 'constraint_violation_mean': float(np.mean(violation_penalties)), 'state_tracking_loss_mean': float(np.mean(state_tracking_losses)), 'power_mean': float(np.mean(powers)), 'peak_power_max_mean': float(np.mean(peaks)), 'startup_extra_power_mean': float(np.mean(startup_extras)), 'total_energy_mean': float(np.mean(total_energies)), 'peak_violation_rate_mean': float(np.mean(peak_violation_rates)) if peak_violation_rates else 0.0, 'avg_power_violation_mean': float(np.mean(avg_power_violation_hist)) if avg_power_violation_hist else 0.0, 'energy_violation_mean': float(np.mean(energy_violation_hist)) if energy_violation_hist else 0.0, 'coverage_mean': float(np.mean(coverages))}
        pd.DataFrame([metrics]).to_csv(run_dir / 'metrics_estimation.csv', index=False)
        if all_actions:
            plot_action_hist(all_actions, run_dir / 'fig_action_hist.png')
        return {'run_dir': str(run_dir), 'metrics': metrics, 'scheduler': name}
    state_dim = len(flatten_rl_state(_current_rl_state(env, estimator, current_event=False)))
    agent = _build_rl_agent(name, state_dim, scheduler_cfg, selector, episode_len=int(run_cfg.get('episode_len', 0)) or None)
    if isinstance(agent, PPOPolicyAdapter):
        raise RuntimeError('PPO should have been handled before the generic DQN training loop')
    num_episodes = int(run_cfg.get('num_episodes', 80))
    save_every = max(1, int(run_cfg.get('save_every', 10)))
    rewards = []
    task_rewards = []
    task_losses = []
    losses = []
    trace_means = []
    uncertainty_means = []
    forecast_means = []
    switch_penalty_means = []
    coverage_penalty_means = []
    violation_penalty_means = []
    state_tracking_means = []
    power_means = []
    peak_means = []
    total_energies = []
    peak_violation_rates = []
    coverage_means = []
    lambda_avg_hist = []
    lambda_energy_hist = []
    avg_power_violation_hist = []
    energy_violation_hist = []
    val_objective_hist: list[float] = []
    val_reward_hist: list[float] = []
    val_forecast_hist: list[float] = []
    val_power_hist: list[float] = []
    val_peak_violation_hist: list[float] = []
    best_val_objective = float('-inf')
    best_val_summary: dict[str, float] | None = None
    last_ckpt = run_dir / f"{_checkpoint_name(name).removesuffix('.pt')}_last.pt"
    best_ckpt = run_dir / _checkpoint_name(name)
    for _ in range(num_episodes):
        reset_out = env.reset()
        estimator.reset()
        current_event = bool(reset_out.get('event_flags', {}).get('event', False))
        estimate_history: list[np.ndarray] = [estimator.get_state_estimate().copy()]
        observed_mask_history: list[np.ndarray] = [np.ones(len(meta['state_columns']), dtype=float)]
        time_index_history: list[int] = [int(env.get_absolute_time_index()) if hasattr(env, 'get_absolute_time_index') else int(env.get_time_index())]
        initial_trace = float(estimator.get_uncertainty_summary()['trace_P'])
        trace_context_history: list[float] = [initial_trace]
        power_context_history: list[float] = [0.0]
        peak_power_context_history: list[float] = [0.0]
        event_context_history: list[int] = [1 if current_event else 0]
        col_to_idx = {name: i for i, name in enumerate(meta['state_columns'])}
        ep_reward = 0.0
        ep_task_reward = 0.0
        ep_losses = []
        ep_trace = []
        ep_uncertainty = []
        ep_forecast = []
        ep_task_loss = []
        ep_switch_penalty = []
        ep_coverage_penalty = []
        ep_violation_penalty = []
        ep_state_tracking = []
        ep_power = []
        ep_peak = []
        ep_startup_extra = []
        ep_cov = []
        prev_selected: list[str] = []
        while True:
            state = _current_rl_state(env, estimator, current_event=current_event)
            state_vec = np.asarray(flatten_rl_state(state), dtype=np.float32)
            raw_action = _agent_act(agent, state_vec, selector, prev_selected, greedy=False)
            selected, aid = _resolve_action(raw_action, selector, prev_selected=prev_selected, scheduler_name=name)
            power_info = selector.power_metrics(selected, prev_selected=prev_selected)
            step = env.step(selected)
            estimator.predict()
            estimator.update(step['available_observations'])
            observed_sensor_ids = [str(obs['sensor_id']) for obs in step['available_observations'] if obs.get('available', False)]
            power_cost = float(power_info['steady_power'])
            power_ratio = power_cost / max(float(getattr(selector, 'per_step_budget', power_cost or 1.0)), 1e-06)
            estimator.on_step(selected_sensor_ids=selected, power_ratio=power_ratio, observed_sensor_ids=observed_sensor_ids)
            current_event = bool(step.get('event_flags', {}).get('event', False))
            next_state = _current_rl_state(env, estimator, current_event=current_event)
            next_vec = np.asarray(flatten_rl_state(next_state), dtype=np.float32)
            unc_summary = estimator.get_uncertainty_summary()
            trace_p = float(unc_summary['trace_P'])
            estimate_history.append(estimator.get_state_estimate().copy())
            state_tracking_loss = 0.0
            if float(task_reward_cfg.get('lambda_state_tracking', 0.0)) > 0.0:
                state_tracking_loss = _state_tracking_loss(estimator, latent_state=step['latent_state'], state_columns=meta['state_columns'], reward_target_indices=reward_target_indices)
            mask = np.zeros(len(meta['state_columns']), dtype=float)
            for obs in step['available_observations']:
                for var_name in obs.get('variables', []):
                    idx = col_to_idx.get(var_name)
                    if idx is not None:
                        mask[idx] = 1.0
            observed_mask_history.append(mask)
            time_index_history.append(int(env.get_absolute_time_index()) if hasattr(env, 'get_absolute_time_index') else int(env.get_time_index()))
            trace_context_history.append(trace_p)
            power_context_history.append(power_cost)
            peak_power_context_history.append(float(power_info['peak_power']))
            event_context_history.append(1 if current_event else 0)
            forecast_loss = _forecast_reward_loss(env, reward_oracle, estimate_history, observed_mask_history, time_index_history, trace_context_history, power_context_history, peak_power_context_history, event_context_history)
            reward_terms = compute_forecast_task_terms(
                forecast_loss=forecast_loss,
                switch_count=_switch_count(prev_selected, selected),
                coverage_ratio=next_state.get('coverage_ratio', []),
                steady_power=power_cost,
                peak_power=float(power_info['peak_power']),
                steady_limit=max(float(getattr(selector, 'per_step_budget', power_cost or 1.0)) - float(getattr(selector, 'safety_margin', 0.0)), 0.0),
                peak_limit=None if getattr(selector, 'startup_peak_budget', None) is None else max(float(selector.startup_peak_budget) - float(getattr(selector, 'safety_margin', 0.0)), 0.0),
                reward_cfg=task_reward_cfg,
                state_tracking_loss=state_tracking_loss,
            )
            task_reward = float(reward_terms['task_reward'])
            if isinstance(agent, (ConstrainedDQNAgent, ConstrainedScoreDQNAgent)):
                reward = float(agent.shape_reward(task_reward=task_reward, steady_power=power_cost))
            else:
                reward = float(task_reward)
            if isinstance(agent, (ScoreDQNAgent, ConstrainedScoreDQNAgent)):
                if isinstance(agent, ConstrainedScoreDQNAgent):
                    info = agent.observe(state_vec, selected, task_reward, next_vec, bool(step['done']), constraint_cost=power_cost)
                else:
                    info = agent.observe(state_vec, selected, reward, next_vec, bool(step['done']))
            else:
                if aid is None:
                    raise ValueError('Discrete RL agent requires a resolved action id')
                if isinstance(agent, ConstrainedDQNAgent):
                    info = agent.observe(state_vec, int(aid), task_reward, next_vec, bool(step['done']), constraint_cost=power_cost)
                else:
                    info = agent.observe(state_vec, int(aid), reward, next_vec, bool(step['done']))
            ep_reward += reward
            ep_task_reward += task_reward
            ep_trace.append(trace_p)
            ep_uncertainty.append(float(unc_summary.get('weighted_trace_P_norm', trace_p)))
            ep_forecast.append(float(forecast_loss))
            ep_task_loss.append(float(reward_terms['task_loss']))
            ep_switch_penalty.append(float(reward_terms['switch_penalty_raw']))
            ep_coverage_penalty.append(float(reward_terms['coverage_penalty_raw']))
            ep_violation_penalty.append(float(reward_terms['violation_penalty_raw']))
            ep_state_tracking.append(float(reward_terms['state_tracking_loss']))
            ep_power.append(power_cost)
            ep_peak.append(float(power_info['peak_power']))
            ep_startup_extra.append(float(power_info['startup_extra_power']))
            cov = next_state.get('coverage_ratio', [])
            ep_cov.append(float(np.mean(cov)) if cov else 0.0)
            loss_value = info.get('loss')
            if loss_value is not None:
                ep_losses.append(float(loss_value))
            prev_selected = list(selected)
            if step['done']:
                break
        constraint_metrics = summarize_constraint_metrics(steady_power_hist=ep_power, peak_power_hist=ep_peak, startup_extra_hist=ep_startup_extra, average_power_budget=constraint_budgets['average_power_budget'], episode_energy_budget=constraint_budgets['episode_energy_budget'], peak_power_budget=constraint_budgets['peak_power_budget'])
        if isinstance(agent, (ConstrainedDQNAgent, ConstrainedScoreDQNAgent)):
            dual_metrics = agent.end_episode(mean_power=float(constraint_metrics['power_mean']), total_energy=float(constraint_metrics['total_energy']))
        else:
            dual_metrics = {'lambda_avg': 0.0, 'lambda_energy': 0.0, 'avg_power_violation': float(constraint_metrics['avg_power_violation']), 'energy_violation': float(constraint_metrics['energy_violation'])}
        rewards.append(ep_reward)
        task_rewards.append(ep_task_reward)
        task_losses.append(float(np.mean(ep_task_loss)))
        trace_means.append(float(np.mean(ep_trace)))
        uncertainty_means.append(float(np.mean(ep_uncertainty)))
        forecast_means.append(float(np.mean(ep_forecast)))
        switch_penalty_means.append(float(np.mean(ep_switch_penalty)))
        coverage_penalty_means.append(float(np.mean(ep_coverage_penalty)))
        violation_penalty_means.append(float(np.mean(ep_violation_penalty)))
        state_tracking_means.append(float(np.mean(ep_state_tracking)))
        power_means.append(float(constraint_metrics['power_mean']))
        peak_means.append(float(constraint_metrics['peak_power_max']))
        total_energies.append(float(constraint_metrics['total_energy']))
        peak_violation_rates.append(float(constraint_metrics['peak_violation_rate']))
        coverage_means.append(float(np.mean(ep_cov)))
        lambda_avg_hist.append(float(dual_metrics.get('lambda_avg', 0.0)))
        lambda_energy_hist.append(float(dual_metrics.get('lambda_energy', 0.0)))
        avg_power_violation_hist.append(float(dual_metrics.get('avg_power_violation', 0.0)))
        energy_violation_hist.append(float(dual_metrics.get('energy_violation', 0.0)))
        if ep_losses:
            losses.append(float(np.mean(ep_losses)))
        val_objective_hist.append(float('nan'))
        val_reward_hist.append(float('nan'))
        val_forecast_hist.append(float('nan'))
        val_power_hist.append(float('nan'))
        val_peak_violation_hist.append(float('nan'))
        if len(rewards) % save_every == 0 or len(rewards) == num_episodes:
            val_summary = _evaluate_agent_on_split(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, scheduler_cfg_path=scheduler_cfg_path, split_name='rl_val', agent=agent, reward_artifact=reward_artifact, base_cfg_path=base_cfg_path)
            val_objective = _validation_objective(val_summary, constrained=isinstance(agent, (ConstrainedDQNAgent, ConstrainedScoreDQNAgent)))
            val_objective_hist[-1] = float(val_objective)
            val_reward_hist[-1] = float(val_summary['task_reward_mean'])
            val_forecast_hist[-1] = float(val_summary['forecast_loss_mean'])
            val_power_hist[-1] = float(val_summary['power_mean'])
            val_peak_violation_hist[-1] = float(val_summary['peak_violation_rate'])
            agent.save(str(last_ckpt))
            if val_objective > best_val_objective:
                best_val_objective = float(val_objective)
                best_val_summary = dict(val_summary)
                agent.save(str(best_ckpt))
    metrics = {'scheduler': name, 'reward_mean': float(np.mean(rewards)), 'task_reward_mean': float(np.mean(task_rewards)), 'task_loss_mean': float(np.mean(task_losses)), 'trace_P_mean': float(np.mean(trace_means)), 'uncertainty_mean': float(np.mean(uncertainty_means)), 'forecast_loss_mean': float(np.mean(forecast_means)), 'switch_penalty_mean': float(np.mean(switch_penalty_means)), 'coverage_penalty_mean': float(np.mean(coverage_penalty_means)), 'constraint_violation_mean': float(np.mean(violation_penalty_means)), 'state_tracking_loss_mean': float(np.mean(state_tracking_means)), 'power_mean': float(np.mean(power_means)), 'peak_power_max_mean': float(np.mean(peak_means)), 'total_energy_mean': float(np.mean(total_energies)), 'peak_violation_rate_mean': float(np.mean(peak_violation_rates)), 'coverage_mean': float(np.mean(coverage_means)), 'lambda_avg_final': float(lambda_avg_hist[-1]) if lambda_avg_hist else 0.0, 'lambda_energy_final': float(lambda_energy_hist[-1]) if lambda_energy_hist else 0.0, 'avg_power_violation_mean': float(np.mean(avg_power_violation_hist)), 'energy_violation_mean': float(np.mean(energy_violation_hist)), 'best_val_objective': float(best_val_objective) if best_val_objective > float('-inf') else float('nan')}
    if best_val_summary is not None:
        metrics.update({'best_val_reward': float(best_val_summary['task_reward_mean']), 'best_val_forecast_loss': float(best_val_summary['forecast_loss_mean']), 'best_val_power_mean': float(best_val_summary['power_mean']), 'best_val_peak_violation_rate': float(best_val_summary['peak_violation_rate'])})
    pd.DataFrame([metrics]).to_csv(run_dir / 'metrics_estimation.csv', index=False)
    pd.DataFrame({'reward': rewards, 'task_reward': task_rewards, 'task_loss': task_losses, 'trace_P': trace_means, 'uncertainty': uncertainty_means, 'forecast_loss': forecast_means, 'switch_penalty': switch_penalty_means, 'coverage_penalty': coverage_penalty_means, 'constraint_violation': violation_penalty_means, 'state_tracking_loss': state_tracking_means, 'power': power_means, 'peak_power_max': peak_means, 'total_energy': total_energies, 'peak_violation_rate': peak_violation_rates, 'coverage': coverage_means, 'lambda_avg': lambda_avg_hist, 'lambda_energy': lambda_energy_hist, 'avg_power_violation': avg_power_violation_hist, 'energy_violation': energy_violation_hist, 'val_objective': val_objective_hist, 'val_reward': val_reward_hist, 'val_forecast_loss': val_forecast_hist, 'val_power': val_power_hist, 'val_peak_violation_rate': val_peak_violation_hist}).to_csv(run_dir / 'training_log.csv', index=False)
    if best_ckpt.exists():
        agent.load(str(best_ckpt))
        ckpt = best_ckpt
    else:
        ckpt = run_dir / _checkpoint_name(name)
        agent.save(str(ckpt))
    plot_training_curves(rewards, losses, run_dir / 'fig_training_curves.png')
    plot_trace_power(trace_means, power_means, run_dir / 'fig_trace_power.png')
    return {'run_dir': str(run_dir), 'metrics': metrics, 'scheduler': name, 'checkpoint': str(ckpt)}

def evaluate_scheduler(truth_csv: str, env_cfg_path: str, sensor_cfg_path: str, estimator_cfg_path: str, scheduler_cfg_path: str, run_id: str, checkpoint: str | None=None, reward_artifact: str | None=None, base_cfg_path: str=DEFAULT_BASE_CFG_PATH) -> dict:
    seed = base_cfg_seed(base_cfg_path)
    env, estimator, selector, meta = _build_truth_stack(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, split_name='final_test', seed=seed, random_reset=False, episode_len=meta_length_from_truth_csv(truth_csv, env_cfg_path, split_name='final_test', base_cfg_path=base_cfg_path), base_cfg_path=base_cfg_path)
    base_cfg = meta['base_cfg']
    scheduler_cfg = load_yaml(scheduler_cfg_path)
    scheduler, name = _make_scheduler(scheduler_cfg, selector, meta['sensor_cfg'], meta['state_columns'])
    run_dir = _build_run_dir(run_id)
    if _is_rl_scheduler(name):
        state_dim = len(flatten_rl_state(_current_rl_state(env, estimator, current_event=False)))
        agent = _build_rl_agent(name, state_dim, scheduler_cfg, selector, episode_len=meta_length_from_truth_csv(truth_csv, env_cfg_path, split_name='final_test', base_cfg_path=base_cfg_path))
        ckpt = checkpoint or run_dir / _checkpoint_name(name)
        agent.load(str(ckpt))

        class _Greedy:

            def reset(self):
                return None

            def act(self, state):
                vec = np.asarray(flatten_rl_state(state), dtype=np.float32)
                prev_selected = _prev_selected_from_state(state, selector)
                return _agent_act(agent, vec, selector, prev_selected, greedy=True)
        scheduler = _Greedy()
    constraints_cfg = base_cfg.get('constraints', {})
    constraint_budgets = _resolve_constraint_budgets(constraints_cfg, scheduler_cfg)
    task_reward_cfg = _task_reward_cfg(base_cfg)
    reward_oracle = _load_active_reward_oracle(base_cfg, reward_artifact)
    out = _rollout_scheduler(env, estimator, scheduler, selector, task_reward_cfg, reward_target_indices=list(meta.get('reward_target_indices', [])), reward_oracle=reward_oracle, greedy=True, scheduler_name=name)
    constraint_metrics = summarize_constraint_metrics(steady_power_hist=out['power_hist'], peak_power_hist=out['peak_power_hist'], startup_extra_hist=out['startup_extra_hist'], average_power_budget=constraint_budgets['average_power_budget'], episode_energy_budget=constraint_budgets['episode_energy_budget'], peak_power_budget=constraint_budgets['peak_power_budget'])
    summary = {'scheduler': name, 'reward_mean': float(out['episode_reward']), 'task_reward_mean': float(out['episode_reward']), 'task_loss_mean': float(np.mean(out['task_loss_hist'])) if out['task_loss_hist'] else float('nan'), 'trace_P_mean': float(np.mean(out['trace_hist'])) if out['trace_hist'] else float('nan'), 'uncertainty_mean': float(np.mean(out['uncertainty_hist'])) if out['uncertainty_hist'] else float('nan'), 'forecast_loss_mean': float(np.mean(out['forecast_hist'])) if out['forecast_hist'] else float('nan'), 'switch_penalty_mean': float(np.mean(out['switch_penalty_hist'])) if out['switch_penalty_hist'] else float('nan'), 'coverage_penalty_mean': float(np.mean(out['coverage_penalty_hist'])) if out['coverage_penalty_hist'] else float('nan'), 'constraint_violation_mean': float(np.mean(out['violation_penalty_hist'])) if out['violation_penalty_hist'] else float('nan'), 'state_tracking_loss_mean': float(np.mean(out['state_tracking_hist'])) if out['state_tracking_hist'] else float('nan'), 'power_mean': float(constraint_metrics['power_mean']), 'peak_power_max': float(constraint_metrics['peak_power_max']), 'startup_extra_power_mean': float(constraint_metrics['startup_extra_power_mean']), 'total_energy': float(constraint_metrics['total_energy']), 'peak_violation_rate': float(constraint_metrics['peak_violation_rate']), 'avg_power_violation': float(constraint_metrics['avg_power_violation']), 'energy_violation': float(constraint_metrics['energy_violation']), 'coverage_mean': float(np.mean(out['coverage_hist'])) if out['coverage_hist'] else float('nan')}
    pd.DataFrame([summary]).to_csv(run_dir / 'metrics_estimation_eval.csv', index=False)
    pd.DataFrame([summary]).to_csv(run_dir / 'metrics_estimation.csv', index=False)
    if out['action_ids']:
        plot_action_hist(out['action_ids'], run_dir / 'fig_action_hist_eval.png')
    plot_trace_power(out['trace_hist'], out['power_hist'], run_dir / 'fig_trace_power_eval.png')
    return {'run_dir': str(run_dir), 'summary': summary}

def meta_length_from_truth_csv(truth_csv: str, env_cfg_path: str, split_name: str, base_cfg_path: str=DEFAULT_BASE_CFG_PATH) -> int:
    truth_df = pd.read_csv(truth_csv)
    env_cfg = load_yaml(env_cfg_path)
    truth_df, _ = _ensure_event_column(truth_df, env_cfg)
    base_cfg = load_yaml(base_cfg_path)
    bounds = _split_bounds(n_rows=len(truth_df), split_cfg=base_cfg.get('data', {}))
    start, end = bounds[split_name]
    return max(end - start, 1)

def build_scheduler_dataset(truth_csv: str, env_cfg_path: str, sensor_cfg_path: str, estimator_cfg_path: str, scheduler_cfg_path: str, run_id: str, out_npz: str, checkpoint: str | None=None, split_name: str='final_test', base_cfg_path: str=DEFAULT_BASE_CFG_PATH) -> dict:
    seed = base_cfg_seed(base_cfg_path)
    env, estimator, selector, meta = _build_truth_stack(truth_csv=truth_csv, env_cfg_path=env_cfg_path, sensor_cfg_path=sensor_cfg_path, estimator_cfg_path=estimator_cfg_path, split_name=split_name, seed=seed, random_reset=False, episode_len=meta_length_from_truth_csv(truth_csv, env_cfg_path, split_name=split_name, base_cfg_path=base_cfg_path), base_cfg_path=base_cfg_path)
    scheduler_cfg = load_yaml(scheduler_cfg_path)
    scheduler, name = _make_scheduler(scheduler_cfg, selector, meta['sensor_cfg'], meta['state_columns'])
    run_dir = _build_run_dir(run_id)
    if _is_rl_scheduler(name):
        state_dim = len(flatten_rl_state(_current_rl_state(env, estimator, current_event=False)))
        agent = _build_rl_agent(name, state_dim, scheduler_cfg, selector)
        ckpt = checkpoint or Path('reports/runs') / run_id / _checkpoint_name(name)
        agent.load(str(ckpt))

        class _Greedy:

            def reset(self):
                return None

            def act(self, state):
                vec = np.asarray(flatten_rl_state(state), dtype=np.float32)
                prev_selected = _prev_selected_from_state(state, selector)
                return _agent_act(agent, vec, selector, prev_selected, greedy=True)
        scheduler = _Greedy()
    base_cfg = meta['base_cfg']
    constraints_cfg = base_cfg.get('constraints', {})
    constraint_budgets = _resolve_constraint_budgets(constraints_cfg, scheduler_cfg)
    task_reward_cfg = _task_reward_cfg(base_cfg)
    rollout = _rollout_scheduler(env, estimator, scheduler, selector, reward_cfg=task_reward_cfg, reward_target_indices=list(meta.get('reward_target_indices', [])), reward_oracle=None, greedy=True, collect_series=True, scheduler_name=name)
    truth_series = np.asarray(rollout['truth_hist'], dtype=float)
    input_series = np.asarray(rollout['estimate_hist'], dtype=float)
    observed_mask = np.asarray(rollout['observed_mask_hist'], dtype=float)
    time_index = np.asarray(rollout['time_index_hist'], dtype=int)
    event_flags = np.asarray(rollout['event_hist'], dtype=int)
    power = np.asarray(rollout['power_hist'], dtype=float)
    peak_power = np.asarray(rollout['peak_power_hist'], dtype=float)
    trace_p = np.asarray(rollout['trace_hist'], dtype=float)
    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, input_series=input_series, target_series=truth_series, observed_mask=observed_mask, event_flags=event_flags, power=power, peak_power=peak_power, trace_p=trace_p, time_index=time_index, feature_names=np.asarray(meta['state_columns']))
    total_full_power = sum((float(s.get('power_cost', 1.0)) for s in meta['sensor_cfg'].get('sensors', [])))
    constraint_metrics = summarize_constraint_metrics(steady_power_hist=rollout['power_hist'], peak_power_hist=rollout['peak_power_hist'], startup_extra_hist=rollout['startup_extra_hist'], average_power_budget=constraint_budgets['average_power_budget'], episode_energy_budget=constraint_budgets['episode_energy_budget'], peak_power_budget=constraint_budgets['peak_power_budget'])
    dataset_meta = {'run_id': run_id, 'scheduler_name': name, 'truth_csv': truth_csv, 'feature_names': meta['state_columns'], 'n_steps': int(input_series.shape[0]), 'dataset_split': str(split_name), 'reward_target_columns': list(meta.get('reward_target_columns', [])), 'forecast_target_columns': list(meta.get('forecast_target_columns', [])), 'base_freq_s': int(meta['env_cfg'].get('base_freq_s', 1)), 'avg_power': float(constraint_metrics['power_mean']), 'total_power': float(constraint_metrics['total_energy']), 'peak_power_max': float(constraint_metrics['peak_power_max']), 'peak_violation_rate': float(constraint_metrics['peak_violation_rate']), 'coverage_mean': float(np.mean(rollout['coverage_hist'])) if rollout['coverage_hist'] else 0.0, 'trace_P_mean': float(np.mean(trace_p)) if trace_p.size else float('nan'), 'uncertainty_mean': float(np.mean(rollout['uncertainty_hist'])) if rollout['uncertainty_hist'] else float('nan'), 'full_open_power': float(total_full_power), 'budget_per_step': float(base_cfg.get('constraints', {}).get('per_step_budget', 0.0)), 'startup_peak_budget': constraint_budgets['peak_power_budget'], 'average_power_budget': constraint_budgets['average_power_budget'], 'episode_energy_budget': constraint_budgets['episode_energy_budget'], 'max_active': int(base_cfg.get('constraints', {}).get('max_active', 0))}
    save_yaml(dataset_meta, out_path.with_suffix('.meta.yaml'))
    pd.DataFrame([dataset_meta]).to_csv(run_dir / 'dataset_stats.csv', index=False)
    return {'run_dir': str(run_dir), 'out_npz': str(out_path), 'meta': dataset_meta}
