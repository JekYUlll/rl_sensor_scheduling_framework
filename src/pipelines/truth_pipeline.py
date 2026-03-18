from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_yaml, save_yaml
from core.seed import set_seed
from envs.truth_replay_env import TruthReplayConfig, TruthReplayEnvironment
from estimators.kalman_filter import KalmanFilterEstimator
from estimators.state_summary import flatten_rl_state
from evaluation.cost_metrics import compute_step_cost
from scheduling.action_space import DiscreteActionSpace
from scheduling.baselines.full_open_scheduler import FullOpenScheduler
from scheduling.baselines.info_priority_scheduler import InfoPriorityScheduler
from scheduling.baselines.periodic_scheduler import PeriodicScheduler
from scheduling.baselines.random_scheduler import RandomScheduler
from scheduling.baselines.round_robin_scheduler import RoundRobinScheduler
from scheduling.rl.dqn_agent import DQNAgent
from sensors.base_sensor import SensorSpec
from sensors.dataset_sensor import DatasetSensor
from visualization.estimation_plots import plot_trace_power
from visualization.policy_plots import plot_action_hist
from visualization.training_curves import plot_training_curves


def _build_run_dir(run_id: str) -> Path:
    root = Path("reports/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _split_bounds(n_rows: int, train_ratio: float, val_ratio: float) -> dict[str, tuple[int, int]]:
    n_train = int(n_rows * train_ratio)
    n_val = int(n_rows * val_ratio)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_rows),
        "all": (0, n_rows),
    }


def _infer_state_columns(df: pd.DataFrame, env_cfg: dict) -> list[str]:
    configured = env_cfg.get("state_columns")
    if configured:
        return [str(col) for col in configured]
    exclude = {"t", "time_idx", "timestamp", "storm_flag", "event_flag"}
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(str(col))
    return numeric_cols


def _ensure_event_column(df: pd.DataFrame, env_cfg: dict) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    event_col = str(env_cfg.get("event_column", "event_flag"))
    if event_col in out.columns:
        out[event_col] = out[event_col].astype(bool)
        return out, event_col
    if "storm_flag" in out.columns:
        out[event_col] = out["storm_flag"].astype(bool)
        return out, event_col

    source_col = str(env_cfg.get("event_source_column", "snow_mass_flux_kg_m2_s"))
    quantile = float(env_cfg.get("event_quantile", 0.9))
    threshold = float(out[source_col].quantile(quantile))
    out[event_col] = out[source_col] >= threshold
    return out, event_col


def _build_sensors(sensor_cfg: dict, state_columns: list[str]) -> list[DatasetSensor]:
    sensors: list[DatasetSensor] = []
    for item in sensor_cfg.get("sensors", []):
        variables = [str(v) for v in item.get("variables", [])]
        noise_std = item.get("noise_std", 0.0)
        if isinstance(noise_std, list):
            obs_dim = len(noise_std)
        else:
            obs_dim = len(variables)
        spec = SensorSpec(
            sensor_id=str(item["sensor_id"]),
            obs_dim=obs_dim,
            variables=variables,
            refresh_interval=int(item.get("refresh_interval", 1)),
            power_cost=float(item.get("power_cost", 1.0)),
            startup_delay=int(item.get("startup_delay", 0)),
            noise_std=noise_std,
        )
        sensors.append(DatasetSensor(spec=spec, state_columns=state_columns))
    return sensors


def _sensor_to_dims(sensor_cfg: dict, state_columns: list[str]) -> dict[str, list[int]]:
    state_to_idx = {name: i for i, name in enumerate(state_columns)}
    out: dict[str, list[int]] = {}
    for sensor in sensor_cfg.get("sensors", []):
        dims = [state_to_idx[v] for v in sensor.get("variables", []) if v in state_to_idx]
        out[str(sensor["sensor_id"])] = dims
    return out


def _make_action_space(sensor_cfg: dict, base_cfg: dict) -> DiscreteActionSpace:
    sensor_ids = [str(s["sensor_id"]) for s in sensor_cfg.get("sensors", [])]
    power_costs = {str(s["sensor_id"]): float(s.get("power_cost", 1.0)) for s in sensor_cfg.get("sensors", [])}
    constraints = base_cfg.get("constraints", {})
    return DiscreteActionSpace(
        sensor_ids=sensor_ids,
        power_costs=power_costs,
        max_active=int(constraints.get("max_active", 2)),
        per_step_budget=float(constraints.get("per_step_budget", 2.0)),
    )


def _build_estimator(
    truth_df: pd.DataFrame,
    state_columns: list[str],
    estimator_cfg: dict,
    sensor_cfg: dict,
    x0: np.ndarray | None = None,
) -> KalmanFilterEstimator:
    values = truth_df[state_columns].to_numpy(dtype=float)
    diffs = np.diff(values, axis=0)
    if diffs.shape[0] >= 2:
        q_mat = np.cov(diffs, rowvar=False)
        if np.ndim(q_mat) == 0:
            q_mat = np.array([[float(q_mat)]], dtype=float)
    else:
        q_mat = np.eye(len(state_columns), dtype=float) * 1e-3
    q_mat = np.asarray(q_mat, dtype=float)
    q_mat += 1e-6 * np.eye(q_mat.shape[0], dtype=float)

    p0_diag = estimator_cfg.get("P0_diag", [1.0])
    if len(p0_diag) == 1:
        p0 = np.diag(np.full(len(state_columns), float(p0_diag[0]), dtype=float))
    elif len(p0_diag) != len(state_columns):
        p0 = np.diag(np.full(len(state_columns), float(p0_diag[0]), dtype=float))
    else:
        p0 = np.diag(np.asarray(p0_diag, dtype=float))

    sensor_ids = [str(s["sensor_id"]) for s in sensor_cfg.get("sensors", [])]
    return KalmanFilterEstimator(
        A=np.eye(len(state_columns), dtype=float),
        Q=q_mat,
        x0=values[0] if x0 is None else np.asarray(x0, dtype=float),
        P0=p0,
        sensor_ids=sensor_ids,
        use_logdet=bool(estimator_cfg.get("use_logdet", False)),
    )


def _make_scheduler(
    scheduler_cfg: dict,
    action_space: DiscreteActionSpace,
    sensor_cfg: dict,
    state_columns: list[str],
):
    name = str(scheduler_cfg.get("scheduler_name", "random"))
    sensor_ids = [str(s["sensor_id"]) for s in sensor_cfg.get("sensors", [])]
    max_active = action_space.max_active
    if name == "random":
        return RandomScheduler(action_space), name
    if name == "periodic":
        return PeriodicScheduler(action_space, period=int(scheduler_cfg.get("period", 1))), name
    if name == "round_robin":
        return RoundRobinScheduler(
            action_space,
            sensor_ids=sensor_ids,
            max_active=max_active,
            min_on_steps=int(scheduler_cfg.get("min_on_steps", 1)),
        ), name
    if name == "info_priority":
        return InfoPriorityScheduler(
            action_space,
            sensor_ids=sensor_ids,
            sensor_to_dims=_sensor_to_dims(sensor_cfg, state_columns),
            max_active=max_active,
            weights=scheduler_cfg.get("weights", {}),
        ), name
    if name == "full_open":
        return FullOpenScheduler(sensor_ids), name
    if name == "dqn":
        return None, name
    raise ValueError(f"Unknown scheduler_name: {name}")


def _resolve_action(action, action_space: DiscreteActionSpace) -> tuple[list[str], int | None]:
    if isinstance(action, (list, tuple)):
        selected = [str(sid) for sid in action]
        return selected, None
    action_id = int(action)
    return action_space.decode(action_id), action_id


def _switch_count(prev_selected: list[str], selected: list[str]) -> int:
    return len(set(prev_selected) ^ set(selected))


def _build_truth_stack(
    truth_csv: str,
    env_cfg_path: str,
    sensor_cfg_path: str,
    estimator_cfg_path: str,
    split_name: str,
    seed: int,
    random_reset: bool,
    episode_len: int | None = None,
):
    set_seed(seed)
    env_cfg = load_yaml(env_cfg_path)
    sensor_cfg = load_yaml(sensor_cfg_path)
    estimator_cfg = load_yaml(estimator_cfg_path)
    base_cfg = load_yaml("configs/base.yaml")

    truth_df = pd.read_csv(truth_csv)
    truth_df, event_col = _ensure_event_column(truth_df, env_cfg)
    state_columns = _infer_state_columns(truth_df, env_cfg)
    split_cfg = base_cfg.get("data", {})
    bounds = _split_bounds(
        n_rows=len(truth_df),
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
    )
    split_start, split_end = bounds[split_name]
    sensors = _build_sensors(sensor_cfg, state_columns)
    run_cfg = base_cfg.get("run", {})
    env_episode_len = episode_len or min(int(run_cfg.get("episode_len", 400)), max(split_end - split_start, 1))

    env = TruthReplayEnvironment(
        truth_df=truth_df,
        sensors=sensors,
        cfg=TruthReplayConfig(
            state_columns=state_columns,
            split_start=split_start,
            split_end=split_end,
            episode_len=env_episode_len,
            random_reset=random_reset,
            base_freq_s=int(env_cfg.get("base_freq_s", 1)),
            event_column=event_col,
        ),
        seed=seed,
    )
    estimator = _build_estimator(
        truth_df.iloc[bounds["train"][0] : bounds["train"][1]].reset_index(drop=True),
        state_columns=state_columns,
        estimator_cfg=estimator_cfg,
        sensor_cfg=sensor_cfg,
        x0=truth_df.iloc[split_start][state_columns].to_numpy(dtype=float),
    )
    action_space = _make_action_space(sensor_cfg, base_cfg)
    meta = {
        "truth_df": truth_df,
        "state_columns": state_columns,
        "event_column": event_col,
        "bounds": bounds,
        "base_cfg": base_cfg,
        "sensor_cfg": sensor_cfg,
    }
    return env, estimator, action_space, meta


def _rollout_scheduler(
    env,
    estimator,
    scheduler,
    action_space: DiscreteActionSpace,
    cost_cfg: dict,
    greedy: bool = False,
    collect_series: bool = False,
):
    reset_out = env.reset()
    estimator.reset()
    scheduler.reset()
    current_event = bool(reset_out.get("event_flags", {}).get("event", False))

    total_reward = 0.0
    action_ids: list[int] = []
    trace_hist: list[float] = []
    power_hist: list[float] = []
    coverage_hist: list[float] = []
    truth_hist: list[list[float]] = []
    estimate_hist: list[list[float]] = []
    observed_mask_hist: list[list[float]] = []
    event_hist: list[int] = []
    prev_selected: list[str] = []
    state_columns = getattr(env, "state_columns", [])
    col_to_idx = {name: i for i, name in enumerate(state_columns)}

    while True:
        state = estimator.get_rl_state_features()
        state["event"] = current_event
        raw_action = scheduler.act({**state, "t": env.get_time_index()}) if greedy else scheduler.act(state)
        selected, action_id = _resolve_action(raw_action, action_space)
        if action_id is not None:
            action_ids.append(action_id)

        step = env.step(selected)
        estimator.predict()
        estimator.update(step["available_observations"])

        observed_sensor_ids = [str(obs["sensor_id"]) for obs in step["available_observations"] if obs.get("available", False)]
        power_cost = float(step.get("info", {}).get("power_cost", 0.0))
        power_ratio = power_cost / max(action_space.per_step_budget, 1e-6)
        estimator.on_step(
            selected_sensor_ids=selected,
            power_ratio=power_ratio,
            observed_sensor_ids=observed_sensor_ids,
        )

        next_state = estimator.get_rl_state_features()
        current_event = bool(step.get("event_flags", {}).get("event", False))
        next_state["event"] = current_event
        trace_p = float(estimator.get_uncertainty_summary()["trace_P"])
        cost = compute_step_cost(
            uncertainty_trace=trace_p,
            power_cost=power_cost,
            switch_count=_switch_count(prev_selected, selected),
            coverage_ratio=next_state.get("coverage_ratio", []),
            cost_cfg=cost_cfg,
        )
        reward = -cost
        total_reward += reward
        trace_hist.append(trace_p)
        power_hist.append(power_cost)
        cov = next_state.get("coverage_ratio", [])
        coverage_hist.append(float(np.mean(cov)) if cov else 0.0)

        if collect_series:
            truth_hist.append([float(step["latent_state"][col]) for col in state_columns])
            estimate_hist.append([float(v) for v in estimator.get_state_estimate().tolist()])
            mask = np.zeros(len(state_columns), dtype=float)
            for obs in step["available_observations"]:
                for var_name in obs.get("variables", []):
                    idx = col_to_idx.get(var_name)
                    if idx is not None:
                        mask[idx] = 1.0
            observed_mask_hist.append(mask.tolist())
            event_hist.append(1 if bool(step.get("event_flags", {}).get("event", False)) else 0)

        prev_selected = list(selected)
        if step["done"]:
            break

    return {
        "episode_reward": total_reward,
        "trace_hist": trace_hist,
        "power_hist": power_hist,
        "coverage_hist": coverage_hist,
        "action_ids": action_ids,
        "truth_hist": truth_hist,
        "estimate_hist": estimate_hist,
        "observed_mask_hist": observed_mask_hist,
        "event_hist": event_hist,
    }


def base_cfg_seed() -> int:
    base = load_yaml("configs/base.yaml")
    return int(base.get("seed", 42))


def run_scheduler_training(
    truth_csv: str,
    env_cfg_path: str,
    sensor_cfg_path: str,
    estimator_cfg_path: str,
    scheduler_cfg_path: str,
    run_id: str,
) -> dict:
    seed = base_cfg_seed()
    env, estimator, action_space, meta = _build_truth_stack(
        truth_csv=truth_csv,
        env_cfg_path=env_cfg_path,
        sensor_cfg_path=sensor_cfg_path,
        estimator_cfg_path=estimator_cfg_path,
        split_name="train",
        seed=seed,
        random_reset=True,
    )
    base_cfg = meta["base_cfg"]
    scheduler_cfg = load_yaml(scheduler_cfg_path)
    scheduler, name = _make_scheduler(scheduler_cfg, action_space, meta["sensor_cfg"], meta["state_columns"])
    run_dir = _build_run_dir(run_id)
    save_yaml(
        {
            "truth_csv": truth_csv,
            "env_cfg": env_cfg_path,
            "sensor_cfg": sensor_cfg_path,
            "estimator_cfg": estimator_cfg_path,
            "scheduler_cfg": scheduler_cfg_path,
        },
        run_dir / "config_used.yaml",
    )

    run_cfg = base_cfg.get("run", {})
    cost_cfg = {
        **base_cfg.get("cost", {}),
        "min_coverage_ratio": float(base_cfg.get("constraints", {}).get("min_coverage_ratio", 0.0)),
    }

    if name != "dqn":
        episodes = int(run_cfg.get("eval_episodes", 10))
        rewards = []
        traces = []
        powers = []
        coverages = []
        all_actions = []
        for _ in range(episodes):
            out = _rollout_scheduler(env, estimator, scheduler, action_space, cost_cfg, greedy=True)
            rewards.append(out["episode_reward"])
            traces.append(float(np.mean(out["trace_hist"])))
            powers.append(float(np.mean(out["power_hist"])))
            coverages.append(float(np.mean(out["coverage_hist"])))
            all_actions.extend(out["action_ids"])

        metrics = {
            "scheduler": name,
            "reward_mean": float(np.mean(rewards)),
            "trace_P_mean": float(np.mean(traces)),
            "power_mean": float(np.mean(powers)),
            "coverage_mean": float(np.mean(coverages)),
        }
        pd.DataFrame([metrics]).to_csv(run_dir / "metrics_estimation.csv", index=False)
        if all_actions:
            plot_action_hist(all_actions, run_dir / "fig_action_hist.png")
        return {"run_dir": str(run_dir), "metrics": metrics, "scheduler": name}

    state_dim = len(flatten_rl_state({**estimator.get_rl_state_features(), "event": False}))
    action_dim = action_space.size()
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=scheduler_cfg)

    num_episodes = int(run_cfg.get("num_episodes", 80))
    rewards = []
    losses = []
    trace_means = []
    power_means = []
    coverage_means = []
    for _ in range(num_episodes):
        reset_out = env.reset()
        estimator.reset()
        current_event = bool(reset_out.get("event_flags", {}).get("event", False))
        ep_reward = 0.0
        ep_losses = []
        ep_trace = []
        ep_power = []
        ep_cov = []
        prev_selected: list[str] = []
        while True:
            state = estimator.get_rl_state_features()
            state["event"] = current_event
            state_vec = np.asarray(flatten_rl_state(state), dtype=np.float32)
            aid = agent.act(state_vec)
            selected, _ = _resolve_action(aid, action_space)

            step = env.step(selected)
            estimator.predict()
            estimator.update(step["available_observations"])
            observed_sensor_ids = [str(obs["sensor_id"]) for obs in step["available_observations"] if obs.get("available", False)]
            power_cost = float(step.get("info", {}).get("power_cost", 0.0))
            power_ratio = power_cost / max(action_space.per_step_budget, 1e-6)
            estimator.on_step(
                selected_sensor_ids=selected,
                power_ratio=power_ratio,
                observed_sensor_ids=observed_sensor_ids,
            )

            next_state = estimator.get_rl_state_features()
            current_event = bool(step.get("event_flags", {}).get("event", False))
            next_state["event"] = current_event
            next_vec = np.asarray(flatten_rl_state(next_state), dtype=np.float32)

            trace_p = float(estimator.get_uncertainty_summary()["trace_P"])
            cost = compute_step_cost(
                uncertainty_trace=trace_p,
                power_cost=power_cost,
                switch_count=_switch_count(prev_selected, selected),
                coverage_ratio=next_state.get("coverage_ratio", []),
                cost_cfg=cost_cfg,
            )
            reward = -cost
            info = agent.observe(state_vec, aid, reward, next_vec, bool(step["done"]))

            ep_reward += reward
            ep_trace.append(trace_p)
            ep_power.append(power_cost)
            cov = next_state.get("coverage_ratio", [])
            ep_cov.append(float(np.mean(cov)) if cov else 0.0)
            if info.get("loss") is not None:
                ep_losses.append(float(info["loss"]))
            prev_selected = list(selected)

            if step["done"]:
                break

        rewards.append(ep_reward)
        trace_means.append(float(np.mean(ep_trace)))
        power_means.append(float(np.mean(ep_power)))
        coverage_means.append(float(np.mean(ep_cov)))
        if ep_losses:
            losses.append(float(np.mean(ep_losses)))

    metrics = {
        "scheduler": "dqn",
        "reward_mean": float(np.mean(rewards)),
        "trace_P_mean": float(np.mean(trace_means)),
        "power_mean": float(np.mean(power_means)),
        "coverage_mean": float(np.mean(coverage_means)),
    }
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics_estimation.csv", index=False)
    pd.DataFrame(
        {
            "reward": rewards,
            "trace_P": trace_means,
            "power": power_means,
            "coverage": coverage_means,
        }
    ).to_csv(run_dir / "training_log.csv", index=False)
    ckpt = run_dir / "scheduler_dqn.pt"
    agent.save(str(ckpt))
    plot_training_curves(rewards, losses, run_dir / "fig_training_curves.png")
    plot_trace_power(trace_means, power_means, run_dir / "fig_trace_power.png")
    return {"run_dir": str(run_dir), "metrics": metrics, "scheduler": "dqn", "checkpoint": str(ckpt)}


def evaluate_scheduler(
    truth_csv: str,
    env_cfg_path: str,
    sensor_cfg_path: str,
    estimator_cfg_path: str,
    scheduler_cfg_path: str,
    run_id: str,
    checkpoint: str | None = None,
) -> dict:
    seed = base_cfg_seed()
    env, estimator, action_space, meta = _build_truth_stack(
        truth_csv=truth_csv,
        env_cfg_path=env_cfg_path,
        sensor_cfg_path=sensor_cfg_path,
        estimator_cfg_path=estimator_cfg_path,
        split_name="test",
        seed=seed,
        random_reset=False,
        episode_len=meta_length_from_truth_csv(truth_csv, env_cfg_path, split_name="test"),
    )
    base_cfg = meta["base_cfg"]
    scheduler_cfg = load_yaml(scheduler_cfg_path)
    scheduler, name = _make_scheduler(scheduler_cfg, action_space, meta["sensor_cfg"], meta["state_columns"])
    run_dir = _build_run_dir(run_id)

    if name == "dqn":
        state_dim = len(flatten_rl_state({**estimator.get_rl_state_features(), "event": False}))
        agent = DQNAgent(state_dim, action_space.size(), scheduler_cfg)
        ckpt = checkpoint or (run_dir / "scheduler_dqn.pt")
        agent.load(str(ckpt))

        class _Greedy:
            def reset(self):
                return None

            def act(self, state):
                vec = np.asarray(flatten_rl_state(state), dtype=np.float32)
                return agent.act(vec, greedy=True)

        scheduler = _Greedy()

    cost_cfg = {
        **base_cfg.get("cost", {}),
        "min_coverage_ratio": float(base_cfg.get("constraints", {}).get("min_coverage_ratio", 0.0)),
    }
    out = _rollout_scheduler(env, estimator, scheduler, action_space, cost_cfg, greedy=True)
    summary = {
        "scheduler": name,
        "reward_mean": float(out["episode_reward"]),
        "trace_P_mean": float(np.mean(out["trace_hist"])) if out["trace_hist"] else float("nan"),
        "power_mean": float(np.mean(out["power_hist"])) if out["power_hist"] else float("nan"),
        "coverage_mean": float(np.mean(out["coverage_hist"])) if out["coverage_hist"] else float("nan"),
    }
    pd.DataFrame([summary]).to_csv(run_dir / "metrics_estimation_eval.csv", index=False)
    pd.DataFrame([summary]).to_csv(run_dir / "metrics_estimation.csv", index=False)
    if out["action_ids"]:
        plot_action_hist(out["action_ids"], run_dir / "fig_action_hist_eval.png")
    plot_trace_power(out["trace_hist"], out["power_hist"], run_dir / "fig_trace_power_eval.png")
    return {"run_dir": str(run_dir), "summary": summary}


def meta_length_from_truth_csv(truth_csv: str, env_cfg_path: str, split_name: str) -> int:
    truth_df = pd.read_csv(truth_csv)
    env_cfg = load_yaml(env_cfg_path)
    truth_df, _ = _ensure_event_column(truth_df, env_cfg)
    base_cfg = load_yaml("configs/base.yaml")
    bounds = _split_bounds(
        n_rows=len(truth_df),
        train_ratio=float(base_cfg.get("data", {}).get("train_ratio", 0.7)),
        val_ratio=float(base_cfg.get("data", {}).get("val_ratio", 0.15)),
    )
    start, end = bounds[split_name]
    return max(end - start, 1)


def build_scheduler_dataset(
    truth_csv: str,
    env_cfg_path: str,
    sensor_cfg_path: str,
    estimator_cfg_path: str,
    scheduler_cfg_path: str,
    run_id: str,
    out_npz: str,
    checkpoint: str | None = None,
) -> dict:
    seed = base_cfg_seed()
    env, estimator, action_space, meta = _build_truth_stack(
        truth_csv=truth_csv,
        env_cfg_path=env_cfg_path,
        sensor_cfg_path=sensor_cfg_path,
        estimator_cfg_path=estimator_cfg_path,
        split_name="all",
        seed=seed,
        random_reset=False,
        episode_len=meta_length_from_truth_csv(truth_csv, env_cfg_path, split_name="all"),
    )
    scheduler_cfg = load_yaml(scheduler_cfg_path)
    scheduler, name = _make_scheduler(scheduler_cfg, action_space, meta["sensor_cfg"], meta["state_columns"])
    run_dir = _build_run_dir(run_id)

    if name == "dqn":
        state_dim = len(flatten_rl_state({**estimator.get_rl_state_features(), "event": False}))
        agent = DQNAgent(state_dim, action_space.size(), scheduler_cfg)
        ckpt = checkpoint or (Path("reports/runs") / run_id / "scheduler_dqn.pt")
        agent.load(str(ckpt))

        class _Greedy:
            def reset(self):
                return None

            def act(self, state):
                vec = np.asarray(flatten_rl_state(state), dtype=np.float32)
                return agent.act(vec, greedy=True)

        scheduler = _Greedy()

    base_cfg = meta["base_cfg"]
    cost_cfg = {
        **base_cfg.get("cost", {}),
        "min_coverage_ratio": float(base_cfg.get("constraints", {}).get("min_coverage_ratio", 0.0)),
    }
    rollout = _rollout_scheduler(
        env,
        estimator,
        scheduler,
        action_space,
        cost_cfg=cost_cfg,
        greedy=True,
        collect_series=True,
    )

    truth_series = np.asarray(rollout["truth_hist"], dtype=float)
    input_series = np.asarray(rollout["estimate_hist"], dtype=float)
    observed_mask = np.asarray(rollout["observed_mask_hist"], dtype=float)
    event_flags = np.asarray(rollout["event_hist"], dtype=int)
    power = np.asarray(rollout["power_hist"], dtype=float)
    trace_p = np.asarray(rollout["trace_hist"], dtype=float)

    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        input_series=input_series,
        target_series=truth_series,
        observed_mask=observed_mask,
        event_flags=event_flags,
        power=power,
        trace_p=trace_p,
        feature_names=np.asarray(meta["state_columns"]),
    )

    total_full_power = sum(float(s.get("power_cost", 1.0)) for s in meta["sensor_cfg"].get("sensors", []))
    dataset_meta = {
        "run_id": run_id,
        "scheduler_name": name,
        "truth_csv": truth_csv,
        "feature_names": meta["state_columns"],
        "n_steps": int(input_series.shape[0]),
        "avg_power": float(np.mean(power)) if power.size else 0.0,
        "total_power": float(np.sum(power)),
        "coverage_mean": float(np.mean(rollout["coverage_hist"])) if rollout["coverage_hist"] else 0.0,
        "trace_P_mean": float(np.mean(trace_p)) if trace_p.size else float("nan"),
        "full_open_power": float(total_full_power),
        "budget_per_step": float(base_cfg.get("constraints", {}).get("per_step_budget", 0.0)),
        "max_active": int(base_cfg.get("constraints", {}).get("max_active", 0)),
    }
    save_yaml(dataset_meta, out_path.with_suffix(".meta.yaml"))
    pd.DataFrame([dataset_meta]).to_csv(run_dir / "dataset_stats.csv", index=False)
    return {"run_dir": str(run_dir), "out_npz": str(out_path), "meta": dataset_meta}
