from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_yaml, save_yaml
from core.seed import set_seed
from envs.linear_gaussian_env import LinearGaussianEnvironment
from estimators.kalman_filter import KalmanFilterEstimator
from estimators.state_summary import flatten_rl_state
from evaluation.cost_metrics import compute_step_cost
from scheduling.action_space import DiscreteActionSpace
from scheduling.baselines.info_priority_scheduler import InfoPriorityScheduler
from scheduling.baselines.max_uncertainty_scheduler import MaxUncertaintyScheduler
from scheduling.baselines.periodic_scheduler import PeriodicScheduler
from scheduling.baselines.random_scheduler import RandomScheduler
from scheduling.baselines.round_robin_scheduler import RoundRobinScheduler
from scheduling.rl.dqn_agent import DQNAgent
from visualization.estimation_plots import plot_trace_power
from visualization.policy_plots import plot_action_hist
from visualization.training_curves import plot_training_curves


def _sensor_to_dims(sensor_cfg: dict) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for s in sensor_cfg.get("sensors", []):
        dims = []
        for v in s.get("variables", []):
            if isinstance(v, str) and v.startswith("x") and v[1:].isdigit():
                dims.append(int(v[1:]))
        out[s["sensor_id"]] = dims
    return out


def build_linear_stack(env_cfg_path: str, sensor_cfg_path: str, estimator_cfg_path: str, seed: int = 42):
    set_seed(seed)
    env_cfg = load_yaml(env_cfg_path)
    sensor_cfg = load_yaml(sensor_cfg_path)
    est_cfg = load_yaml(estimator_cfg_path)

    env = LinearGaussianEnvironment.from_dict(env_cfg=env_cfg, sensor_cfg=sensor_cfg)
    sensor_ids = [s["sensor_id"] for s in sensor_cfg.get("sensors", [])]
    power_costs = {s["sensor_id"]: float(s.get("power_cost", 1.0)) for s in sensor_cfg.get("sensors", [])}

    base_cfg = load_yaml("configs/base.yaml")
    constraints = base_cfg.get("constraints", {})
    action_space = DiscreteActionSpace(
        sensor_ids=sensor_ids,
        power_costs=power_costs,
        max_active=int(constraints.get("max_active", 2)),
        per_step_budget=float(constraints.get("per_step_budget", 2.0)),
    )

    P0_diag = np.asarray(est_cfg.get("P0_diag", [1.0] * env.state_dim), dtype=float)
    estimator = KalmanFilterEstimator(
        A=env.cfg.A,
        Q=env.cfg.Q,
        x0=env.cfg.x0,
        P0=np.diag(P0_diag),
        sensor_ids=sensor_ids,
        use_logdet=bool(est_cfg.get("use_logdet", False)),
    )
    return env, estimator, action_space, base_cfg, sensor_cfg


def make_scheduler(scheduler_cfg: dict, action_space, sensor_cfg: dict):
    name = scheduler_cfg.get("scheduler_name", "random")
    sensor_ids = [s["sensor_id"] for s in sensor_cfg.get("sensors", [])]
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
    if name == "max_uncertainty":
        return MaxUncertaintyScheduler(action_space, sensor_ids=sensor_ids, sensor_to_dims=_sensor_to_dims(sensor_cfg), max_active=max_active), name
    if name == "info_priority":
        return InfoPriorityScheduler(
            action_space,
            sensor_ids=sensor_ids,
            sensor_to_dims=_sensor_to_dims(sensor_cfg),
            max_active=max_active,
            weights=scheduler_cfg.get("weights", {}),
        ), name
    if name == "dqn":
        return None, name
    raise ValueError(f"Unknown scheduler_name: {name}")


def _build_run_dir(run_id: str) -> Path:
    root = Path("reports/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _rollout_baseline(env, estimator, scheduler, action_space, cost_cfg: dict):
    env.reset()
    estimator.reset()
    scheduler.reset()

    total_reward = 0.0
    action_ids = []
    trace_hist = []
    power_hist = []
    coverage_hist = []

    while True:
        state = estimator.get_rl_state_features()
        aid = int(scheduler.act({**state, "t": env.get_time_index()}))
        selected = action_space.decode(aid)
        action_ids.append(aid)

        step = env.step(selected)
        estimator.predict()
        estimator.update(step["available_observations"])

        power_cost = float(step.get("info", {}).get("power_cost", 0.0))
        power_ratio = power_cost / max(action_space.per_step_budget, 1e-6)
        estimator.on_step(selected, power_ratio=power_ratio)

        trace_p = float(estimator.get_uncertainty_summary()["trace_P"])
        cov = estimator.get_rl_state_features().get("coverage_ratio", [])
        cost = compute_step_cost(
            uncertainty_trace=trace_p,
            power_cost=power_cost,
            switch_count=0,
            coverage_ratio=cov,
            cost_cfg=cost_cfg,
        )
        reward = -cost
        total_reward += reward
        trace_hist.append(trace_p)
        power_hist.append(power_cost)
        coverage_hist.append(float(np.mean(cov)) if cov else 0.0)

        if step["done"]:
            break

    return {
        "episode_reward": total_reward,
        "trace_hist": trace_hist,
        "power_hist": power_hist,
        "coverage_hist": coverage_hist,
        "action_ids": action_ids,
    }


def run_scheduler_training(
    env_cfg_path: str,
    sensor_cfg_path: str,
    estimator_cfg_path: str,
    scheduler_cfg_path: str,
    run_id: str,
) -> dict:
    env, estimator, action_space, base_cfg, sensor_cfg = build_linear_stack(env_cfg_path, sensor_cfg_path, estimator_cfg_path, seed=base_cfg_seed())
    scheduler_cfg = load_yaml(scheduler_cfg_path)
    scheduler, name = make_scheduler(scheduler_cfg, action_space, sensor_cfg)

    run_dir = _build_run_dir(run_id)
    save_yaml(
        {
            "env_cfg": env_cfg_path,
            "sensor_cfg": sensor_cfg_path,
            "estimator_cfg": estimator_cfg_path,
            "scheduler_cfg": scheduler_cfg_path,
        },
        run_dir / "config_used.yaml",
    )

    run_cfg = base_cfg.get("run", {})
    cost_cfg = {**base_cfg.get("cost", {}), "min_coverage_ratio": float(base_cfg.get("constraints", {}).get("min_coverage_ratio", 0.0))}

    if name != "dqn":
        episodes = int(run_cfg.get("eval_episodes", 10))
        rewards = []
        traces = []
        powers = []
        all_actions = []
        for _ in range(episodes):
            out = _rollout_baseline(env, estimator, scheduler, action_space, cost_cfg)
            rewards.append(out["episode_reward"])
            traces.append(float(np.mean(out["trace_hist"])))
            powers.append(float(np.mean(out["power_hist"])))
            all_actions.extend(out["action_ids"])

        metrics = {
            "scheduler": name,
            "reward_mean": float(np.mean(rewards)),
            "trace_P_mean": float(np.mean(traces)),
            "power_mean": float(np.mean(powers)),
        }
        pd.DataFrame([metrics]).to_csv(run_dir / "metrics_estimation.csv", index=False)
        plot_action_hist(all_actions, run_dir / "fig_action_hist.png")
        return {"run_dir": str(run_dir), "metrics": metrics, "scheduler": name}

    # DQN training
    env.reset()
    estimator.reset()
    state_dim = len(flatten_rl_state(estimator.get_rl_state_features()))
    action_dim = action_space.size()
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=scheduler_cfg)

    num_episodes = int(run_cfg.get("num_episodes", 80))
    max_steps = int(run_cfg.get("max_steps", 400))

    rewards = []
    losses = []
    trace_means = []
    power_means = []

    for _ in range(num_episodes):
        env.reset()
        estimator.reset()
        ep_reward = 0.0
        ep_losses = []
        ep_trace = []
        ep_power = []
        for _step in range(max_steps):
            state = estimator.get_rl_state_features()
            state_vec = np.asarray(flatten_rl_state(state), dtype=np.float32)
            aid = agent.act(state_vec)
            selected = action_space.decode(aid)

            step = env.step(selected)
            estimator.predict()
            estimator.update(step["available_observations"])

            power_cost = float(step.get("info", {}).get("power_cost", 0.0))
            power_ratio = power_cost / max(action_space.per_step_budget, 1e-6)
            estimator.on_step(selected, power_ratio)

            next_state = estimator.get_rl_state_features()
            next_state["event"] = bool(step.get("event_flags", {}).get("event", False))
            next_vec = np.asarray(flatten_rl_state(next_state), dtype=np.float32)

            trace_p = float(estimator.get_uncertainty_summary()["trace_P"])
            cost = compute_step_cost(
                uncertainty_trace=trace_p,
                power_cost=power_cost,
                switch_count=0,
                coverage_ratio=next_state.get("coverage_ratio", []),
                cost_cfg=cost_cfg,
            )
            reward = -cost
            info = agent.observe(state_vec, aid, reward, next_vec, bool(step["done"]))

            ep_reward += reward
            ep_trace.append(trace_p)
            ep_power.append(power_cost)
            if info.get("loss") is not None:
                ep_losses.append(float(info["loss"]))

            if step["done"]:
                break

        rewards.append(ep_reward)
        trace_means.append(float(np.mean(ep_trace)))
        power_means.append(float(np.mean(ep_power)))
        if ep_losses:
            losses.append(float(np.mean(ep_losses)))

    metrics = {
        "scheduler": "dqn",
        "reward_mean": float(np.mean(rewards)),
        "trace_P_mean": float(np.mean(trace_means)),
        "power_mean": float(np.mean(power_means)),
    }

    pd.DataFrame([metrics]).to_csv(run_dir / "metrics_estimation.csv", index=False)
    pd.DataFrame({"reward": rewards, "trace_P": trace_means, "power": power_means}).to_csv(run_dir / "training_log.csv", index=False)
    ckpt = run_dir / "scheduler_dqn.pt"
    agent.save(str(ckpt))
    plot_training_curves(rewards, losses, run_dir / "fig_training_curves.png")
    plot_trace_power(trace_means, power_means, run_dir / "fig_trace_power.png")

    return {"run_dir": str(run_dir), "metrics": metrics, "scheduler": "dqn", "checkpoint": str(ckpt)}


def base_cfg_seed() -> int:
    base = load_yaml("configs/base.yaml")
    return int(base.get("seed", 42))


def evaluate_scheduler(
    env_cfg_path: str,
    sensor_cfg_path: str,
    estimator_cfg_path: str,
    scheduler_cfg_path: str,
    run_id: str,
    checkpoint: str | None = None,
) -> dict:
    env, estimator, action_space, base_cfg, sensor_cfg = build_linear_stack(env_cfg_path, sensor_cfg_path, estimator_cfg_path, seed=base_cfg_seed())
    scheduler_cfg = load_yaml(scheduler_cfg_path)
    scheduler, name = make_scheduler(scheduler_cfg, action_space, sensor_cfg)
    run_dir = _build_run_dir(run_id)

    if name == "dqn":
        state_dim = len(flatten_rl_state(estimator.get_rl_state_features()))
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

    episodes = int(base_cfg.get("run", {}).get("eval_episodes", 10))
    cost_cfg = {**base_cfg.get("cost", {}), "min_coverage_ratio": float(base_cfg.get("constraints", {}).get("min_coverage_ratio", 0.0))}

    metrics = []
    trace_concat = []
    power_concat = []
    action_concat = []
    for _ in range(episodes):
        out = _rollout_baseline(env, estimator, scheduler, action_space, cost_cfg)
        metrics.append({
            "reward": out["episode_reward"],
            "trace_P": float(np.mean(out["trace_hist"])),
            "power": float(np.mean(out["power_hist"])),
            "coverage": float(np.mean(out["coverage_hist"])),
        })
        trace_concat.extend(out["trace_hist"])
        power_concat.extend(out["power_hist"])
        action_concat.extend(out["action_ids"])

    df = pd.DataFrame(metrics)
    df.to_csv(run_dir / "metrics_estimation_eval.csv", index=False)
    summary = {
        "scheduler": name,
        "reward_mean": float(df["reward"].mean()),
        "trace_P_mean": float(df["trace_P"].mean()),
        "power_mean": float(df["power"].mean()),
        "coverage_mean": float(df["coverage"].mean()),
    }
    pd.DataFrame([summary]).to_csv(run_dir / "metrics_estimation.csv", index=False)
    plot_action_hist(action_concat, run_dir / "fig_action_hist_eval.png")
    plot_trace_power(trace_concat, power_concat, run_dir / "fig_trace_power_eval.png")
    return {"run_dir": str(run_dir), "summary": summary}
