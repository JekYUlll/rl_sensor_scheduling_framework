from __future__ import annotations

import numpy as np
import pandas as pd

from envs.truth_replay_env import TruthReplayConfig, TruthReplayEnvironment
from estimators.kalman_filter import KalmanFilterEstimator
from estimators.state_summary import flatten_rl_state
from sensors.base_sensor import SensorSpec
from sensors.dataset_sensor import DatasetSensor


def test_dataset_sensor_warmup_progression_requires_ready_state() -> None:
    spec = SensorSpec(
        sensor_id="wind",
        obs_dim=1,
        variables=["wind_speed_ms"],
        refresh_interval=1,
        power_cost=1.0,
        warmup_steps=2,
        noise_std=0.0,
    )
    sensor = DatasetSensor(spec=spec, state_columns=["wind_speed_ms"])
    latent = {"wind_speed_ms": 5.0}

    status0 = sensor.begin_step(True, t=0)
    obs0 = sensor.observe(latent, t=0)
    assert status0["mode"] == "warming"
    assert status0["warm_remaining_steps"] == 2
    assert obs0["available"] is False

    status1 = sensor.begin_step(True, t=1)
    obs1 = sensor.observe(latent, t=1)
    assert status1["mode"] == "warming"
    assert status1["warm_remaining_steps"] == 1
    assert obs1["available"] is False

    status2 = sensor.begin_step(True, t=2)
    obs2 = sensor.observe(latent, t=2)
    assert status2["mode"] == "ready"
    assert status2["warm_remaining_steps"] == 0
    assert obs2["available"] is True
    np.testing.assert_allclose(obs2["y"], np.asarray([5.0], dtype=float))


def test_truth_replay_env_charges_power_while_sensor_is_warming() -> None:
    truth_df = pd.DataFrame(
        {
            "wind_speed_ms": [4.0, 5.0, 6.0, 7.0],
            "event_flag": [0, 0, 0, 0],
            "time_idx": [0, 1, 2, 3],
        }
    )
    sensor = DatasetSensor(
        spec=SensorSpec(
            sensor_id="wind",
            obs_dim=1,
            variables=["wind_speed_ms"],
            refresh_interval=1,
            power_cost=1.5,
            warmup_steps=2,
            noise_std=0.0,
        ),
        state_columns=["wind_speed_ms"],
    )
    env = TruthReplayEnvironment(
        truth_df=truth_df,
        sensors=[sensor],
        cfg=TruthReplayConfig(
            state_columns=["wind_speed_ms"],
            split_start=0,
            split_end=len(truth_df),
            episode_len=len(truth_df),
            random_reset=False,
        ),
        seed=42,
    )
    env.reset()

    step1 = env.step(["wind"])
    assert step1["available_observations"] == []
    assert step1["warming_sensor_ids"] == ["wind"]
    assert step1["ready_sensor_ids"] == []
    assert np.isclose(step1["info"]["power_cost"], 1.5)

    step2 = env.step(["wind"])
    assert step2["available_observations"] == []
    assert step2["warming_sensor_ids"] == ["wind"]
    assert np.isclose(step2["info"]["power_cost"], 1.5)

    step3 = env.step(["wind"])
    assert len(step3["available_observations"]) == 1
    assert step3["warming_sensor_ids"] == []
    assert step3["ready_sensor_ids"] == ["wind"]
    assert np.isclose(step3["info"]["power_cost"], 1.5)


def test_kalman_rl_state_exposes_warmup_features() -> None:
    estimator = KalmanFilterEstimator(
        A=np.eye(1),
        Q=np.eye(1) * 0.01,
        x0=np.asarray([0.0], dtype=float),
        P0=np.eye(1),
        sensor_ids=["s0"],
        sensor_warmup_steps={"s0": 4},
    )

    estimator.on_step(
        selected_sensor_ids=["s0"],
        observed_sensor_ids=[],
        power_ratio=0.5,
        sensor_status={
            "s0": {
                "powered": True,
                "warming": True,
                "ready": False,
                "warm_remaining_steps": 2,
            }
        },
    )

    rl_state = estimator.get_rl_state_features()
    assert rl_state["powered_mask"] == [1.0]
    assert rl_state["warming_mask"] == [1.0]
    assert rl_state["ready_mask"] == [0.0]
    assert rl_state["warm_remaining"] == [2.0]
    assert rl_state["warm_remaining_norm"] == [0.5]

    flat = flatten_rl_state(rl_state)
    assert flat[-4:-1] == [1.0, 0.0, 0.5]
