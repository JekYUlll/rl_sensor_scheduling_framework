from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from v2.dqn import DQNConfig, DQNTrainer, evaluate_dqn  # noqa: E402
from v2.env import WarmupEnvConfig  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle, OracleConfig, build_supervised_windows  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.sensor_spec import SensorSpecV2  # noqa: E402


STATE_COLUMNS = (
    "wind_speed_ms",
    "wind_direction_deg",
    "wind_dir_sin",
    "wind_dir_cos",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
    "solar_radiation_wm2",
    "snow_surface_temperature_c",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
    "snow_mass_flux_kg_m2_s",
)


def _truth(rows: int = 64) -> pd.DataFrame:
    t = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "wind_speed_ms": 8.0 + 0.1 * t,
            "wind_direction_deg": 90.0 + t,
            "wind_dir_sin": np.sin(np.deg2rad(90.0 + t)),
            "wind_dir_cos": np.cos(np.deg2rad(90.0 + t)),
            "air_temperature_c": -20.0 + 0.05 * t,
            "relative_humidity": 65.0,
            "air_pressure_pa": 70000.0,
            "solar_radiation_wm2": 100.0,
            "snow_surface_temperature_c": -21.0,
            "snow_particle_mean_diameter_mm": 0.2,
            "snow_particle_mean_velocity_ms": 2.0,
            "snow_mass_flux_kg_m2_s": 1e-5 * t,
            "event_flag": t > rows // 2,
        }
    )


def _sensors() -> list[SensorSpecV2]:
    return [
        SensorSpecV2("met", ("wind_speed_ms", "air_temperature_c"), 0.5, 0.8, warmup_steps=0),
        SensorSpecV2("snow", ("snow_mass_flux_kg_m2_s",), 1.2, 1.8, warmup_steps=2),
        SensorSpecV2("rad", ("solar_radiation_wm2",), 0.4, 0.5, warmup_steps=0),
    ]


def _oracle(truth: pd.DataFrame) -> LinearFrozenForecastOracle:
    truth_values = truth[list(STATE_COLUMNS)].to_numpy(dtype=float)
    masks = np.ones_like(truth_values)
    x, y = build_supervised_windows(truth_values, masks, truth_values, lookback=4, horizon=2)
    return LinearFrozenForecastOracle(OracleConfig(lookback=4, horizon=2, ridge_alpha=0.1)).fit(x, y)


def test_dqn_short_run_and_eval(tmp_path: Path) -> None:
    truth = _truth(64)
    sensors = _sensors()
    constraints = PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0)
    candidate_masks = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    env_cfg = WarmupEnvConfig(
        state_columns=STATE_COLUMNS,
        reward_target_columns=STATE_COLUMNS,
        reward_proxy_mode="uncertainty",
        lookback=4,
        episode_len=12,
        seed=3,
        min_dwell_steps=3,
    )
    trainer = DQNTrainer(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        env_cfg=env_cfg,
        oracle=_oracle(truth),
        candidate_masks=candidate_masks,
        cfg=DQNConfig(
            total_timesteps=12,
            replay_size=32,
            learning_starts=2,
            batch_size=4,
            train_freq=1,
            target_update_interval=4,
            hidden_dim=16,
            oracle_prefill_steps=4,
            oracle_prefill_lookahead_steps=1,
            device="cpu",
            seed=5,
            log_interval=6,
        ),
    )
    episode_env = trainer._make_env(seed_offset=3)
    assert episode_env.cfg.min_dwell_steps == env_cfg.min_dwell_steps
    assert episode_env.cfg.reward_proxy_mode == env_cfg.reward_proxy_mode
    assert episode_env.cfg.seed == env_cfg.seed + 3

    trainer.train()
    trainer.save(tmp_path / "dqn.pt")
    result, metrics = evaluate_dqn(
        trainer=trainer,
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=8,
            seed=11,
        ),
        oracle=_oracle(truth),
        steps=8,
        start_indices=(0,),
    )

    assert trainer.history
    assert trainer.replay.count >= 4
    assert np.isfinite(float(trainer.history[-1]["loss"]))
    assert result.policy_name == "dqn"
    assert metrics["policy"] == "dqn"
    assert (tmp_path / "dqn.pt").exists()
