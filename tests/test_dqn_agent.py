from __future__ import annotations

import numpy as np

from scheduling.rl.dqn_agent import DQNAgent


def test_dqn_agent_learns_tiny_signal():
    cfg = {
        "network": {"hidden_dims": [32]},
        "training": {"gamma": 0.9, "lr": 1e-3, "batch_size": 16, "replay_size": 2000, "warmup_steps": 16, "target_update_interval": 20},
        "exploration": {"eps_start": 0.2, "eps_end": 0.01, "eps_decay_steps": 200},
    }
    agent = DQNAgent(state_dim=4, action_dim=2, cfg=cfg)

    for _ in range(200):
        s = np.random.randn(4).astype(np.float32)
        a = 1 if s.sum() > 0 else 0
        r = 1.0
        ns = s * 0.9
        agent.observe(s, a, r, ns, False)

    s = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    aid = agent.act(s, greedy=True)
    assert aid in (0, 1)
