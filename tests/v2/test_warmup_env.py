from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv
from v2.oracle import LinearFrozenForecastOracle, OracleConfig, build_supervised_windows
from v2.power_projector import PowerConstraintsV2, PowerProjector
from v2.rollout import run_policy_rollout
from v2.sb3_ppo import collect_oracle_greedy_bc_dataset
from v2.sensor_spec import SensorSpecV2
from v2.warmup_state import SensorMode, SensorRuntime


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


def _truth(rows: int = 16) -> pd.DataFrame:
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
            "event_flag": t > 10,
        }
    )


def _sensors() -> list[SensorSpecV2]:
    return [
        SensorSpecV2("met", ("wind_speed_ms", "air_temperature_c"), 0.5, 0.8, warmup_steps=0),
        SensorSpecV2("snow", ("snow_mass_flux_kg_m2_s",), 1.2, 1.8, warmup_steps=2),
        SensorSpecV2("rad", ("solar_radiation_wm2",), 0.4, 0.5, warmup_steps=0),
    ]


def test_power_projector_respects_steady_and_peak_constraints() -> None:
    sensors = _sensors()
    runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
    projector = PowerProjector(sensors, PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0))

    result = projector.project_scores(np.asarray([0.9, 0.8, 0.7]), runtimes)

    assert result.feasible
    assert result.steady_power <= 1.7
    assert result.peak_power <= 2.0
    assert "met" in result.selected_sensor_ids


def test_power_projector_always_keeps_required_sensors() -> None:
    sensors = _sensors()
    runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
    projector = PowerProjector(
        sensors,
        PowerConstraintsV2(
            max_active=3,
            per_step_budget=1.7,
            startup_peak_budget=2.0,
            required_sensor_ids=("met", "rad"),
        ),
    )

    result = projector.project_scores(np.asarray([-10.0, 10.0, -10.0]), runtimes)

    assert result.feasible
    assert set(result.selected_sensor_ids) == {"met", "rad"}
    assert result.selected_mask.tolist() == [True, False, True]


def test_power_projector_rejects_infeasible_required_sensors() -> None:
    sensors = _sensors()
    runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
    projector = PowerProjector(
        sensors,
        PowerConstraintsV2(
            max_active=1,
            per_step_budget=1.7,
            startup_peak_budget=2.0,
            required_sensor_ids=("met", "rad"),
        ),
    )

    try:
        projector.project_scores(np.asarray([0.0, 0.0, 0.0]), runtimes)
    except ValueError as exc:
        assert "Required sensor" in str(exc)
    else:
        raise AssertionError("Expected infeasible required sensors to raise")


def test_power_projector_satisfies_coverage_groups_before_filling() -> None:
    sensors = _sensors()
    runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
    projector = PowerProjector(
        sensors,
        PowerConstraintsV2(
            max_active=3,
            per_step_budget=1.7,
            startup_peak_budget=2.0,
            coverage_groups=(
                ("weather", ("met",)),
                ("radiation", ("rad",)),
            ),
        ),
    )

    result = projector.project_scores(np.asarray([-10.0, 10.0, -10.0]), runtimes)

    assert result.feasible
    assert "met" in result.selected_sensor_ids
    assert "rad" in result.selected_sensor_ids
    assert "snow" not in result.selected_sensor_ids


def test_power_projector_rejects_infeasible_coverage_group() -> None:
    sensors = _sensors()
    runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
    projector = PowerProjector(
        sensors,
        PowerConstraintsV2(
            max_active=1,
            per_step_budget=1.7,
            startup_peak_budget=2.0,
            coverage_groups=(
                ("weather", ("met",)),
                ("radiation", ("rad",)),
            ),
        ),
    )

    try:
        projector.project_scores(np.asarray([0.0, 0.0, 0.0]), runtimes)
    except ValueError as exc:
        assert "Coverage group" in str(exc)
    else:
        raise AssertionError("Expected infeasible coverage group to raise")


def test_sensor_warmup_requires_consecutive_selection() -> None:
    runtime = SensorRuntime(SensorSpecV2("snow", ("snow_mass_flux_kg_m2_s",), 1.0, 1.5, warmup_steps=2))

    runtime.begin_step(True, step_idx=0)
    assert runtime.mode == SensorMode.WARMING
    assert not runtime.can_observe(0)
    runtime.end_step(True)
    assert runtime.warm_remaining == 1

    runtime.begin_step(False, step_idx=1)
    assert runtime.mode == SensorMode.OFF
    assert runtime.warmup_abort_count == 1


def test_warmup_env_rollout_returns_state_and_info() -> None:
    env = WarmupSchedulingEnv(
        _truth(24),
        _sensors(),
        PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        WarmupEnvConfig(state_columns=STATE_COLUMNS, lookback=4, episode_len=8, seed=1),
    )
    state, info = env.reset()
    assert state.ndim == 1
    assert info["power"] == 0.0
    assert np.allclose(env.history, env.state_mean.reshape(1, -1))
    assert np.all(env.mask_history == 0.0)

    for _ in range(4):
        state, reward, done, info = env.step_scores(np.asarray([1.0, 0.5, 0.2]))
        assert state.ndim == 1
        assert reward <= 0.0
        assert float(info["power"]) <= 1.7
        assert float(info["peak_power"]) <= 2.0
        if done:
            break


def test_sensor_quality_is_online_and_degrades_measurements() -> None:
    truth = _truth(24).copy()
    quality_columns = tuple(f"quality_{sensor.sensor_id}" for sensor in _sensors())
    for column in quality_columns:
        truth[column] = 1.0
    truth.loc[:, quality_columns[0]] = 0.2
    cfg = WarmupEnvConfig(
        state_columns=STATE_COLUMNS,
        lookback=2,
        episode_len=2,
        seed=3,
        include_event_flag_in_state=False,
        sensor_quality_columns=quality_columns,
        sensor_quality_max_noise_multiplier=6.0,
        sensor_quality_availability_floor=0.2,
    )
    env = WarmupSchedulingEnv(
        truth,
        _sensors(),
        PowerConstraintsV2(max_active=3, per_step_budget=2.5, startup_peak_budget=3.1),
        cfg,
    )
    state, _ = env.reset()

    assert np.allclose(state[-3:], [0.2, 1.0, 1.0])
    assert env._sensor_quality_noise_multiplier(_sensors()[0]) == pytest.approx(5.0)


def test_warmup_env_uses_external_normalization_statistics() -> None:
    mean = tuple(0.0 for _ in STATE_COLUMNS)
    std = tuple(1.0 for _ in STATE_COLUMNS)
    env = WarmupSchedulingEnv(
        _truth(24),
        _sensors(),
        PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            lookback=1,
            episode_len=2,
            seed=1,
            normalization_mean=mean,
            normalization_std=std,
        ),
    )

    env.reset()

    assert np.allclose(env.state_mean, np.zeros(len(STATE_COLUMNS)))
    assert np.allclose(env.state_std, np.ones(len(STATE_COLUMNS)))


def test_uncertainty_reward_decreases_when_target_is_observed() -> None:
    cfg = WarmupEnvConfig(
        state_columns=STATE_COLUMNS,
        reward_target_columns=("air_temperature_c",),
        reward_proxy_mode="uncertainty",
        lookback=2,
        episode_len=2,
        seed=5,
        uncertainty_process_variance=tuple(0.05 for _ in STATE_COLUMNS),
    )
    constraints = PowerConstraintsV2(max_active=1, per_step_budget=1.0, startup_peak_budget=1.0)
    observed_env = WarmupSchedulingEnv(_truth(24), _sensors(), constraints, cfg)
    stale_env = WarmupSchedulingEnv(_truth(24), _sensors(), constraints, cfg)
    observed_env.reset()
    stale_env.reset()

    _, _, _, observed_info = observed_env.step_mask(np.asarray([True, False, False]))
    _, _, _, stale_info = stale_env.step_mask(np.asarray([False, False, True]))

    assert float(observed_info["reward_proxy_loss"]) < float(stale_info["reward_proxy_loss"])
    assert str(observed_info["reward_proxy_mode"]) == "uncertainty"


def _single_temperature_env(*, noise_std: float, update_mode: str) -> WarmupSchedulingEnv:
    truth = _truth(24).copy()
    truth.loc[0, "air_temperature_c"] = 10.0
    sensor = SensorSpecV2(
        "temperature",
        ("air_temperature_c",),
        0.5,
        0.5,
        warmup_steps=0,
        noise_std={"air_temperature_c": noise_std},
    )
    cfg = WarmupEnvConfig(
        state_columns=STATE_COLUMNS,
        lookback=1,
        episode_len=1,
        seed=3,
        normalization_mean=tuple(0.0 for _ in STATE_COLUMNS),
        normalization_std=tuple(1.0 for _ in STATE_COLUMNS),
        uncertainty_process_variance=tuple(0.05 for _ in STATE_COLUMNS),
        uncertainty_initial_variance=1.0,
        measurement_update_mode=update_mode,
    )
    return WarmupSchedulingEnv(
        truth,
        [sensor],
        PowerConstraintsV2(max_active=1, per_step_budget=1.0, startup_peak_budget=1.0),
        cfg,
    )


def test_variance_weighted_measurement_update_accepts_exact_measurement() -> None:
    env = _single_temperature_env(noise_std=0.0, update_mode="variance_weighted")
    env.reset()

    env.step_mask(np.asarray([True]))

    idx = env.state_index["air_temperature_c"]
    assert env.last_observation[idx] == pytest.approx(10.0)
    assert env.posterior_variance[idx] < 1.0e-8


def test_variance_weighted_measurement_update_smooths_noisy_measurement() -> None:
    direct_env = _single_temperature_env(noise_std=1.0, update_mode="direct")
    filtered_env = _single_temperature_env(noise_std=1.0, update_mode="variance_weighted")
    direct_env.reset()
    filtered_env.reset()

    direct_env.step_mask(np.asarray([True]))
    filtered_env.step_mask(np.asarray([True]))

    idx = direct_env.state_index["air_temperature_c"]
    expected_gain = 1.05 / (1.05 + 1.0)
    assert filtered_env.last_observation[idx] == pytest.approx(
        expected_gain * direct_env.last_observation[idx]
    )
    assert abs(filtered_env.last_observation[idx]) < abs(direct_env.last_observation[idx])


def test_online_event_context_prefers_observable_alert_proxy() -> None:
    truth = _truth(24).copy()
    truth["event_flag"] = True
    truth["agent_context_event_alert"] = 0.23
    env = WarmupSchedulingEnv(
        truth,
        _sensors(),
        PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            lookback=2,
            episode_len=2,
            agent_context_columns=("agent_context_event_alert",),
            include_event_flag_in_state=True,
        ),
    )
    env.reset()

    assert env.online_event_context() == pytest.approx(0.23)


def test_rollout_truth_is_aligned_with_observation_step() -> None:
    class MetOnlyPolicy:
        name = "met_only"

        def reset(self) -> None:
            pass

        def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
            del env
            return np.asarray([1.0, -1.0, -1.0], dtype=float)

    truth = _truth(8)
    env = WarmupSchedulingEnv(
        truth,
        _sensors(),
        PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        WarmupEnvConfig(state_columns=STATE_COLUMNS, lookback=4, episode_len=4, seed=1),
    )

    result = run_policy_rollout(env, MetOnlyPolicy(), steps=3)

    assert np.allclose(result.truth[0], truth[list(STATE_COLUMNS)].iloc[0].to_numpy(dtype=float))
    assert np.allclose(result.truth[1], truth[list(STATE_COLUMNS)].iloc[1].to_numpy(dtype=float))


def test_oracle_greedy_bc_dataset_collects_valid_discrete_actions() -> None:
    truth = _truth(48)
    truth_values = truth[list(STATE_COLUMNS)].to_numpy(dtype=float)
    masks = np.ones_like(truth_values)
    x, y = build_supervised_windows(truth_values, masks, truth_values, lookback=4, horizon=2)
    oracle = LinearFrozenForecastOracle(
        OracleConfig(lookback=4, horizon=2, ridge_alpha=0.1)
    ).fit(x, y)
    candidate_masks = np.asarray(
        [
            [True, False, False],
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    obs, actions = collect_oracle_greedy_bc_dataset(
        truth_df=truth,
        sensor_specs=_sensors(),
        constraints=PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        cfg=WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=8,
            seed=7,
        ),
        oracle=oracle,
        candidate_masks=candidate_masks,
        total_steps=10,
        n_rollouts=2,
        event_fraction=0.5,
        greedy_lookahead_steps=2,
        seed=11,
    )

    assert obs.shape[0] == actions.shape[0]
    assert obs.shape[0] > 0
    assert np.all(actions >= 0)
    assert np.all(actions < candidate_masks.shape[0])
