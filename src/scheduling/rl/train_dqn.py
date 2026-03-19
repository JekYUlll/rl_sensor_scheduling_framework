from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estimators.state_summary import flatten_rl_state
from evaluation.cost_metrics import compute_step_cost


@dataclass
class TrainOutput:
    rewards: list[float]
    losses: list[float]
    mean_trace_p: list[float]
    mean_power: list[float]


def rollout_episode(env, estimator, scheduler, action_space, cost_cfg: dict, greedy: bool = False):
    env.reset()
    estimator.reset()
    scheduler.reset()

    total_reward = 0.0
    trace_hist = []
    power_hist = []
    steps = 0

    rl_state = estimator.get_rl_state_features()
    state_vec = np.asarray(flatten_rl_state(rl_state), dtype=np.float32)

    while True:
        action_id = scheduler.act({**rl_state, "t": env.get_time_index()}) if greedy else scheduler.act(state_vec)
        selected = action_space.decode(action_id)
        step = env.step(selected)

        estimator.predict()
        estimator.update(step["available_observations"])
        power_cost = float(step.get("info", {}).get("power_cost", 0.0))
        power_ratio = power_cost / max(action_space.per_step_budget, 1e-6)
        estimator.on_step(selected_sensor_ids=selected, power_ratio=power_ratio)

        next_rl_state = estimator.get_rl_state_features()
        next_rl_state["event"] = bool(step.get("event_flags", {}).get("event", step.get("event_flags", {}).get("storm", False)))
        next_state_vec = np.asarray(flatten_rl_state(next_rl_state), dtype=np.float32)

        unc = estimator.get_uncertainty_summary()
        step_cost = compute_step_cost(
            uncertainty_summary=unc,
            power_cost=power_cost,
            switch_count=0,
            coverage_ratio=next_rl_state.get("coverage_ratio", []),
            cost_cfg=cost_cfg,
        )
        reward = -step_cost
        total_reward += reward
        trace_hist.append(float(unc["trace_P"]))
        power_hist.append(power_cost)
        steps += 1

        done = bool(step["done"])
        if not greedy:
            scheduler.observe(state_vec, action_id, reward, next_state_vec, done)

        rl_state = next_rl_state
        state_vec = next_state_vec
        if done:
            break

    return {
        "episode_reward": total_reward,
        "mean_trace_P": float(np.mean(trace_hist)) if trace_hist else float("nan"),
        "mean_power": float(np.mean(power_hist)) if power_hist else 0.0,
        "steps": steps,
    }
