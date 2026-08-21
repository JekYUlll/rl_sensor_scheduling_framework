from __future__ import annotations

import numpy as np
import torch

from scheduling.online_projector import OnlineSubsetProjector
from scheduling.rl.replay_buffer import ConstraintReplayBuffer, ReplayBuffer
from scheduling.rl.score_dqn_agent import ConstrainedScoreDQNAgent, ScoreDQNAgent


class _PreferHeavyMask(torch.nn.Module):
    def forward(self, states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        del states
        return 10.0 * masks[:, 2] + 0.01 * masks.sum(dim=1)


def test_score_dqn_bootstrap_uses_transition_aware_candidate_masks() -> None:
    projector = OnlineSubsetProjector(
        sensor_ids=["cheap_a", "cheap_b", "heavy"],
        power_costs={"cheap_a": 0.2, "cheap_b": 0.2, "heavy": 1.0},
        startup_peak_costs={"cheap_a": 0.2, "cheap_b": 0.2, "heavy": 1.6},
        max_active=2,
        per_step_budget=1.2,
        startup_peak_budget=1.3,
    )
    agent = ScoreDQNAgent(
        state_dim=4,
        sensor_ids=list(projector.sensor_ids),
        cfg={
            "device": "cpu",
            "network": {"hidden_dims": [8]},
            "training": {"batch_size": 2, "warmup_steps": 2},
            "reward_normalization": {"enabled": False},
        },
        projector=projector,
    )

    next_states = torch.zeros((2, 4), dtype=torch.float32)
    prev_action_masks = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    best_masks = agent._best_next_masks(
        next_states,
        prev_action_masks,
        reward_network=_PreferHeavyMask(),
    )

    assert float(best_masks[0, 2]) == 0.0
    assert float(best_masks[1, 2]) == 1.0


def test_cmdp_score_dqn_without_long_horizon_budget_delegates_to_plain_dqn() -> None:
    projector = OnlineSubsetProjector(
        sensor_ids=["cheap_a", "cheap_b"],
        power_costs={"cheap_a": 0.2, "cheap_b": 0.3},
        startup_peak_costs={"cheap_a": 0.2, "cheap_b": 0.3},
        max_active=2,
        per_step_budget=1.0,
    )
    agent = ConstrainedScoreDQNAgent(
        state_dim=3,
        sensor_ids=list(projector.sensor_ids),
        cfg={
            "device": "cpu",
            "network": {"hidden_dims": [8]},
            "training": {"batch_size": 2, "warmup_steps": 2},
            "reward_normalization": {"enabled": False},
            "cmdp": {"average_power_budget": None, "episode_energy_budget": None, "lambda_avg_init": 0.0},
        },
        projector=projector,
    )

    assert agent.constraint_active is False
    assert isinstance(agent.replay, ReplayBuffer)
    assert not hasattr(agent, "cost_q")
    assert agent.shape_reward(task_reward=1.5, steady_power=0.9) == 1.5

    state = np.zeros(3, dtype=np.float32)
    agent.observe(state, ["cheap_a"], 1.0, state, False, constraint_cost=0.2)
    assert len(agent.replay) == 1


def test_cmdp_score_dqn_with_average_budget_activates_dual_layer() -> None:
    projector = OnlineSubsetProjector(
        sensor_ids=["cheap_a", "heavy"],
        power_costs={"cheap_a": 0.2, "heavy": 1.4},
        startup_peak_costs={"cheap_a": 0.2, "heavy": 1.6},
        max_active=2,
        per_step_budget=2.0,
    )
    agent = ConstrainedScoreDQNAgent(
        state_dim=3,
        sensor_ids=list(projector.sensor_ids),
        cfg={
            "device": "cpu",
            "network": {"hidden_dims": [8]},
            "training": {"batch_size": 2, "warmup_steps": 2},
            "reward_normalization": {"enabled": False},
            "cmdp": {
                "average_power_budget": 1.0,
                "lambda_avg_init": 0.0,
                "dual_lr_avg": 0.03,
                "violation_ema_beta": 0.8,
                "power_reference": 2.0,
            },
        },
        projector=projector,
    )

    assert agent.constraint_active is True
    assert isinstance(agent.replay, ConstraintReplayBuffer)
    assert hasattr(agent, "cost_q")
    metrics = agent.end_episode(mean_power=1.6, total_energy=819.2)
    assert metrics["avg_power_violation"] > 0.0
    assert metrics["lambda_cost"] > 0.0
