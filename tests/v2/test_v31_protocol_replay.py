from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_overlapping_random_starts_are_deterministic_and_disjoint() -> None:
    module = _load_script("57_v31_independent_replay.py")
    starts = module.non_overlapping_random_starts(
        sequence_steps=30000,
        eval_steps=1024,
        horizon=8,
        n_rollouts=6,
        seed=11818,
    )
    repeated = module.non_overlapping_random_starts(
        sequence_steps=30000,
        eval_steps=1024,
        horizon=8,
        n_rollouts=6,
        seed=11818,
    )

    assert starts == repeated
    assert len(starts) == 6
    for idx, left in enumerate(starts):
        for right in starts[idx + 1 :]:
            assert left + 1024 <= right or right + 1024 <= left


def test_interval_overlap_pairs_detect_cross_window_overlap() -> None:
    module = _load_script("56_v31_protocol_audit.py")

    pairs = module.interval_overlap_pairs((100, 1000), 200, (250, 1200), 200)

    assert pairs == [(0, 0)]


def test_split_protocol_reserves_non_overlapping_final_test_windows() -> None:
    module = _load_script("58_v31_split_protocol_run.py")
    bounds = module.partition_bounds(90000, (0.35, 0.50, 0.075, 0.075))
    final_starts = module.non_overlapping_starts(
        bounds=bounds["final_test"],
        window_steps=1024,
        horizon=8,
        count=6,
        seed=1818,
    )

    assert bounds["final_test"] == (83250, 90000)
    assert len(final_starts) == 6
    assert min(final_starts) >= bounds["final_test"][0]
    assert max(final_starts) + 1024 + 8 < bounds["final_test"][1]


def test_control_source_files_follow_reward_normalization_mode() -> None:
    module = _load_script("25_v2_train_custom_ppo.py")

    unnormalized = module.control_source_required_files("none")
    staticnorm = module.control_source_required_files("staticnorm_subtype")

    assert "reward_staticnorm_candidates.csv" not in unnormalized
    assert "reward_staticnorm_normalizers.json" not in unnormalized
    assert "reward_staticnorm_candidates.csv" in staticnorm
    assert "reward_staticnorm_normalizers.json" in staticnorm


def test_score_policy_replay_preserves_all_online_state_configuration() -> None:
    module = _load_script("23_v2_train_ppo.py")
    cfg = module.WarmupEnvConfig(
        state_columns=("x",),
        reward_target_columns=("x",),
        lookback=4,
        episode_len=8,
        seed=7,
        sensor_quality_columns=("quality_a", "quality_b"),
        include_alert_context_features=True,
        alert_context_columns=("particle_alert", "flux_alert", "thermal_alert"),
        alert_context_threshold=0.6,
        alert_context_trend_lookback=9,
    )
    replay_cfg = module.rollout_config_with_seed(cfg, offset=3)

    assert replay_cfg.seed == 10
    assert replay_cfg.sensor_quality_columns == cfg.sensor_quality_columns
    assert replay_cfg.include_alert_context_features is True
    assert replay_cfg.alert_context_columns == cfg.alert_context_columns
    assert replay_cfg.alert_context_threshold == 0.6
    assert replay_cfg.alert_context_trend_lookback == 9


def test_receding_command_preserves_alert_context_contract() -> None:
    module = _load_script("99_v32_receding_upper.py")
    command: list[str] = []
    module.append_alert_context_args(
        command,
        {
            "agent_alert_context": {
                "columns": ["particle_alert", "flux_alert", "thermal_alert"],
                "threshold": 0.6,
                "trend_lookback": 9,
                "include_alert_context_features": True,
                "include_event_flag_in_state": False,
            }
        },
    )

    assert command == [
        "--alert-context-columns", "particle_alert", "flux_alert", "thermal_alert",
        "--alert-context-threshold", "0.6",
        "--alert-context-trend-lookback", "9",
        "--include-alert-context-features",
        "--no-include-event-flag-in-state",
    ]


def test_flexible_behavior_gate_preserves_frozen_six_channel_rule() -> None:
    module = _load_script("25_v2_train_custom_ppo.py")
    valid = {
        "always_on_sensor_count": 0,
        "always_off_sensor_count": 1,
        "switches_per_step": 0.01,
        "warmup_abort_count": 0,
    }

    assert module.flexible_six_channel_behavior_valid(valid)
    assert not module.flexible_six_channel_behavior_valid(
        {**valid, "always_on_sensor_count": 1}
    )
    assert not module.flexible_six_channel_behavior_valid(
        {**valid, "always_off_sensor_count": 2}
    )
    assert not module.flexible_six_channel_behavior_valid(
        {**valid, "switches_per_step": 0.0}
    )
    assert not module.flexible_six_channel_behavior_valid(
        {**valid, "warmup_abort_count": 1}
    )


def test_multiscene_collector_uses_same_behavior_gate() -> None:
    module = _load_script("102_v32_collect_multiscene_loo.py")
    import pandas as pd

    valid = pd.Series({
        "always_on_sensor_count": 0,
        "always_off_sensor_count": 1,
        "switches_per_step": 0.01,
        "warmup_abort_count": 0,
    })
    invalid = valid.copy()
    invalid["always_off_sensor_count"] = 2

    assert module.behavior_valid(valid)
    assert not module.behavior_valid(invalid)


def test_channel_quality_generation_is_deterministic_and_bounded() -> None:
    module = _load_script("20_build_public_weather_truth.py")
    import pandas as pd

    frame = pd.DataFrame({"x": range(200)})
    kwargs = dict(
        sensor_ids=("a", "b"),
        seed=17,
        coverage=0.2,
        min_duration=8,
        max_duration=16,
        min_gap=4,
        degraded_quality=0.2,
        transition_steps=0,
        report_noise_std=0.0,
    )
    first = module.add_channel_quality_dynamics(frame, **kwargs)
    second = module.add_channel_quality_dynamics(frame, **kwargs)

    assert first.equals(second)
    assert first["agent_context_quality_a"].between(0.2, 1.0).all()
    assert (first["agent_context_quality_a"] < 1.0).any()
    assert not first["agent_context_quality_a"].equals(first["agent_context_quality_b"])


def test_channel_quality_transition_is_gradual() -> None:
    module = _load_script("20_build_public_weather_truth.py")
    import pandas as pd

    frame = pd.DataFrame({"x": range(240)})
    out = module.add_channel_quality_dynamics(
        frame,
        sensor_ids=("a",),
        seed=29,
        coverage=0.2,
        min_duration=24,
        max_duration=24,
        min_gap=8,
        degraded_quality=0.2,
        transition_steps=6,
        report_noise_std=0.0,
    )
    quality = out["agent_context_quality_a"].to_numpy()

    assert np.any((quality > 0.2) & (quality < 1.0))
    assert np.max(np.abs(np.diff(quality))) < 0.8


def test_condition_dependent_quality_is_bounded_and_exogenous_to_availability() -> None:
    module = _load_script("20_build_public_weather_truth.py")
    import pandas as pd

    frame = pd.DataFrame({
        "wind_speed_ms": [8.0, 10.0, 15.0, 19.0, 21.0],
        "relative_humidity": [60.0, 65.0, 70.0, 78.0, 86.0],
        "event_subtype_particle_latent": [0.0, 0.2, 1.0, 0.0, 0.0],
        "event_subtype_flux_latent": [0.0, 0.0, 0.1, 1.0, 0.0],
        "event_subtype_thermal_latent": [0.0, 0.0, 0.0, 0.0, 1.0],
    })
    out = module.add_channel_quality_dynamics(
        frame,
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=31,
        coverage=0.0,
        min_duration=1,
        max_duration=1,
        min_gap=0,
        degraded_quality=0.35,
        transition_steps=0,
        report_noise_std=0.0,
        mode="condition_dependent",
    )

    for column in out.filter(like="agent_context_quality_"):
        assert out[column].between(0.35, 1.0).all()
    assert out.loc[2, "agent_context_quality_fc4_flux"] < out.loc[2, "agent_context_quality_laser_disdrometer"]
    assert out.loc[3, "agent_context_quality_laser_disdrometer"] < out.loc[3, "agent_context_quality_fc4_flux"]


def test_exposure_recovery_state_is_deterministic_and_replaces_quality_columns() -> None:
    module = _load_script("20_build_public_weather_truth.py")
    import pandas as pd

    rows = 64
    frame = pd.DataFrame({
        "wind_speed_ms": np.linspace(2.0, 15.0, rows),
        "relative_humidity": np.linspace(55.0, 95.0, rows),
        "air_temperature_c": np.linspace(-5.0, -28.0, rows),
        "solar_radiation_wm2": np.linspace(240.0, 5.0, rows),
        "agent_context_nowcast_wind_speed_ms": np.linspace(2.5, 14.5, rows),
        "agent_context_nowcast_relative_humidity": np.linspace(57.0, 93.0, rows),
        "agent_context_nowcast_air_temperature_c": np.linspace(-4.0, -26.0, rows),
        "agent_context_nowcast_solar_radiation_wm2": np.linspace(235.0, 8.0, rows),
        "snow_mass_flux_kg_m2_s": np.full(rows, 0.02),
        "snow_particle_mean_velocity_ms": np.full(rows, 4.0),
        "snow_particle_mean_diameter_mm": np.full(rows, 0.2),
        "snow_surface_temperature_c": np.full(rows, -15.0),
    })
    sensors = ("surface_temp_ir", "laser_disdrometer", "fc4_flux")
    first = module.add_exposure_recovery_dynamics(frame, sensor_ids=sensors)
    second = module.add_exposure_recovery_dynamics(frame, sensor_ids=sensors)

    assert first.equals(second)
    assert {"generator_exposure_transport_state", "generator_exposure_frost_state"}.issubset(first.columns)
    for sensor in sensors:
        for prefix in ("agent_context_quality_", "agent_context_quality_forecast_"):
            assert first[f"{prefix}{sensor}"].between(0.10, 1.0).all()
    assert first["generator_exposure_transport_state"].iloc[-1] > first["generator_exposure_transport_state"].iloc[0]


def test_balanced_exposure_recovery_state_recovers_after_loading() -> None:
    module = _load_script("20_build_public_weather_truth.py")
    import pandas as pd

    rows = 96
    loaded = np.r_[np.linspace(2.0, 15.0, rows // 2), np.linspace(15.0, 2.0, rows // 2)]
    humidity = np.r_[np.linspace(55.0, 95.0, rows // 2), np.linspace(95.0, 40.0, rows // 2)]
    temperature = np.r_[np.linspace(-5.0, -28.0, rows // 2), np.linspace(-28.0, 2.0, rows // 2)]
    solar = np.r_[np.linspace(240.0, 5.0, rows // 2), np.linspace(5.0, 300.0, rows // 2)]
    frame = pd.DataFrame({
        "wind_speed_ms": loaded,
        "relative_humidity": humidity,
        "air_temperature_c": temperature,
        "solar_radiation_wm2": solar,
        "agent_context_nowcast_wind_speed_ms": loaded,
        "agent_context_nowcast_relative_humidity": humidity,
        "agent_context_nowcast_air_temperature_c": temperature,
        "agent_context_nowcast_solar_radiation_wm2": solar,
        "snow_mass_flux_kg_m2_s": np.full(rows, 0.02),
        "snow_particle_mean_velocity_ms": np.full(rows, 4.0),
        "snow_particle_mean_diameter_mm": np.full(rows, 0.2),
        "snow_surface_temperature_c": np.full(rows, -15.0),
    })
    out = module.add_exposure_recovery_dynamics(
        frame,
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        balanced_recovery=True,
    )

    transport = out["generator_exposure_transport_state"]
    frost = out["generator_exposure_frost_state"]
    assert transport.iloc[-1] < transport.iloc[rows // 2]
    assert frost.iloc[-1] < frost.iloc[rows // 2]
    assert transport.between(0.0, 1.0).all()
    assert frost.between(0.0, 1.0).all()


def test_decoupled_exposure_drivers_reduce_state_correlation() -> None:
    module = _load_script("20_build_public_weather_truth.py")
    import pandas as pd

    rows = 192
    phase = np.linspace(0.0, 8.0 * np.pi, rows)
    wind = 8.0 + 5.0 * np.sin(phase)
    humidity = 70.0 + 20.0 * np.cos(phase / 2.0)
    temperature = -15.0 + 10.0 * np.cos(phase / 2.0)
    solar = 150.0 + 130.0 * np.cos(phase / 2.0)
    frame = pd.DataFrame({
        "wind_speed_ms": wind,
        "relative_humidity": humidity,
        "air_temperature_c": temperature,
        "solar_radiation_wm2": solar,
        "agent_context_nowcast_wind_speed_ms": wind,
        "agent_context_nowcast_relative_humidity": humidity,
        "agent_context_nowcast_air_temperature_c": temperature,
        "agent_context_nowcast_solar_radiation_wm2": solar,
        "snow_mass_flux_kg_m2_s": np.full(rows, 0.02),
        "snow_particle_mean_velocity_ms": np.full(rows, 4.0),
        "snow_particle_mean_diameter_mm": np.full(rows, 0.2),
        "snow_surface_temperature_c": np.full(rows, -15.0),
    })
    coupled = module.add_exposure_recovery_dynamics(frame, sensor_ids=(), balanced_recovery=True)
    decoupled = module.add_exposure_recovery_dynamics(
        frame,
        sensor_ids=(),
        balanced_recovery=True,
        decoupled_drivers=True,
    )
    columns = ["generator_exposure_transport_state", "generator_exposure_frost_state"]
    assert abs(decoupled[columns].corr().iloc[0, 1]) < abs(coupled[columns].corr().iloc[0, 1])

    amplified = module.add_exposure_recovery_dynamics(
        frame,
        sensor_ids=(),
        balanced_recovery=True,
        decoupled_drivers=True,
        target_gain=2.0,
    )
    baseline_flux = frame["snow_mass_flux_kg_m2_s"]
    assert (amplified["snow_mass_flux_kg_m2_s"] - baseline_flux).abs().mean() > (
        decoupled["snow_mass_flux_kg_m2_s"] - baseline_flux
    ).abs().mean()

    frame["blowing_snow_active"] = np.r_[np.zeros(rows // 4), np.ones(rows // 2), np.zeros(rows // 4)]
    low_frequency = module.add_exposure_recovery_dynamics(
        frame,
        sensor_ids=(),
        balanced_recovery=True,
        decoupled_drivers=True,
        low_frequency_targets=True,
        residual_fraction=0.35,
    )
    calm = frame["blowing_snow_active"] == 0
    for target in (
        "snow_mass_flux_kg_m2_s",
        "snow_particle_mean_velocity_ms",
        "snow_particle_mean_diameter_mm",
    ):
        assert np.allclose(low_frequency.loc[calm, target], frame.loc[calm, target])

    causal = module.add_exposure_recovery_dynamics(
        frame,
        sensor_ids=(),
        balanced_recovery=True,
        decoupled_drivers=True,
        causal_anomaly_drivers=True,
    )
    prefix = module.add_exposure_recovery_dynamics(
        frame.iloc[: rows // 2].copy(),
        sensor_ids=(),
        balanced_recovery=True,
        decoupled_drivers=True,
        causal_anomaly_drivers=True,
    )
    for state in columns:
        assert np.allclose(causal[state].iloc[: rows // 2], prefix[state])

    three_factor = module.add_three_factor_exposure_dynamics(frame, sensor_ids=())
    three_factor_prefix = module.add_three_factor_exposure_dynamics(
        frame.iloc[: rows // 2].copy(), sensor_ids=()
    )
    for state in (
        "generator_flux_exposure_state",
        "generator_particle_exposure_state",
        "generator_thermal_exposure_state",
    ):
        assert np.allclose(three_factor[state].iloc[: rows // 2], three_factor_prefix[state])

    forecast_value = module.add_forecast_value_dynamics(
        frame, sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"), seed=41
    )
    forecast_value_prefix = module.add_forecast_value_dynamics(
        frame.iloc[: rows // 2].copy(),
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=41,
    )
    for factor in ("flux", "particle", "thermal"):
        state = f"generator_{factor}_demand_state"
        context = f"agent_context_forecast_{factor}_demand"
        assert np.allclose(forecast_value[state].iloc[: rows // 2], forecast_value_prefix[state])
        assert np.allclose(forecast_value[context].iloc[: rows // 2], forecast_value_prefix[context])
        assert forecast_value[state].between(0.0, 1.0).all()
        assert forecast_value[context].between(0.0, 1.0).all()
    for sensor in ("surface_temp_ir", "laser_disdrometer", "fc4_flux"):
        assert forecast_value[f"agent_context_quality_{sensor}"].between(0.72, 1.0).all()
        assert forecast_value[f"agent_context_quality_forecast_{sensor}"].between(0.72, 1.0).all()
    assert forecast_value["snow_surface_temperature_c"].between(-80.0, 10.0).all()

    stationary = module.add_forecast_value_dynamics(
        frame,
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=43,
        stationary_local_state=True,
        forecast_lead_steps=8,
    )
    stationary_prefix = module.add_forecast_value_dynamics(
        frame.iloc[: rows // 2].copy(),
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=43,
        stationary_local_state=True,
        forecast_lead_steps=8,
    )
    for factor in ("flux", "particle", "thermal"):
        state = f"generator_{factor}_demand_state"
        assert np.allclose(stationary[state].iloc[: rows // 2], stationary_prefix[state])
        assert stationary[f"agent_context_forecast_{factor}_demand"].between(0.0, 1.0).all()

    residence = module.add_forecast_value_dynamics(
        frame,
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=47,
        residence_local_state=True,
        forecast_lead_steps=8,
    )
    residence_prefix = module.add_forecast_value_dynamics(
        frame.iloc[: rows // 2].copy(),
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=47,
        residence_local_state=True,
        forecast_lead_steps=8,
    )
    for factor in ("flux", "particle", "thermal"):
        state = f"generator_{factor}_demand_state"
        assert np.allclose(residence[state].iloc[: rows // 2], residence_prefix[state])
        assert residence[state].between(0.05, 0.95).all()
        assert residence[f"agent_context_forecast_{factor}_demand"].between(0.0, 1.0).all()

    persistent = module.add_forecast_value_dynamics(
        frame,
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=47,
        residence_local_state=True,
        horizon_persistent_latent=True,
        forecast_lead_steps=8,
    )
    for target in (
        "snow_surface_temperature_c",
        "snow_mass_flux_kg_m2_s",
        "snow_particle_mean_diameter_mm",
        "snow_particle_mean_velocity_ms",
    ):
        assert np.isfinite(persistent[target]).all()

    resilient = module.add_forecast_value_dynamics(
        frame,
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=47,
        residence_local_state=True,
        horizon_persistent_latent=True,
        specialist_resilient_quality=True,
        forecast_lead_steps=8,
    )
    particle = resilient["generator_particle_demand_state"].to_numpy(dtype=float)
    thermal = resilient["generator_thermal_demand_state"].to_numpy(dtype=float)
    assert np.allclose(
        resilient["agent_context_quality_surface_temp_ir"],
        np.clip(0.97 - 0.03 * particle, 0.72, 1.0),
    )
    assert np.allclose(
        resilient["agent_context_quality_laser_disdrometer"],
        np.clip(0.97 - 0.03 * thermal, 0.72, 1.0),
    )
    assert np.allclose(
        resilient["agent_context_quality_fc4_flux"],
        np.clip(0.97 - 0.03 * particle, 0.72, 1.0),
    )

    activity_aligned = module.add_forecast_value_dynamics(
        frame,
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=47,
        residence_local_state=True,
        horizon_persistent_latent=True,
        specialist_resilient_quality=True,
        activity_aligned_transport_demand=True,
        forecast_lead_steps=8,
    )
    active = frame["blowing_snow_active"].to_numpy(dtype=bool)
    for factor in ("flux", "particle"):
        effective = activity_aligned[f"generator_{factor}_demand_state"].to_numpy(dtype=float)
        raw = activity_aligned[f"generator_{factor}_demand_state_raw"].to_numpy(dtype=float)
        assert np.all(effective[~active] == 0.0)
        assert np.allclose(effective[active], raw[active])
        assert np.all((raw >= 0.05) & (raw <= 0.95))
        assert activity_aligned[f"agent_context_forecast_{factor}_demand"].between(0.0, 1.0).all()
    assert np.allclose(
        activity_aligned["generator_thermal_demand_state"],
        resilient["generator_thermal_demand_state"],
    )

    with pytest.raises(ValueError, match="requires residence local state"):
        module.add_forecast_value_dynamics(
            frame,
            sensor_ids=("laser_disdrometer", "fc4_flux"),
            seed=47,
            activity_aligned_transport_demand=True,
        )


def test_subset_geometry_recognizes_forecast_demand_states() -> None:
    import pandas as pd

    module = _load_script("109_v32_audit_subset_forecast_geometry.py")
    truth = pd.DataFrame({
        "generator_flux_demand_state": [0.1, 0.9, 0.1, 0.9],
        "generator_particle_demand_state": [0.1, 0.1, 0.9, 0.9],
        "generator_thermal_demand_state": [0.1, 0.9, 0.9, 0.1],
    })
    meta = {"partition_protocol": {"normalization_start_idx": 0, "normalization_end_idx": 4}}

    labels, columns, thresholds = module.operating_condition_labels(truth, meta)

    assert columns == (
        "generator_flux_demand_state",
        "generator_particle_demand_state",
        "generator_thermal_demand_state",
    )
    assert len(set(labels)) == 4
    assert all(value == 0.5 for value in thresholds.values())


def test_subset_geometry_activity_aligned_thresholds_ignore_inactive_zeros() -> None:
    import pandas as pd

    module = _load_script("109_v32_audit_subset_forecast_geometry.py")
    truth = pd.DataFrame({
        "generator_flux_demand_state": [0.0, 0.0, 0.2, 0.8],
        "generator_particle_demand_state": [0.0, 0.0, 0.4, 0.6],
        "generator_thermal_demand_state": [0.1, 0.2, 0.8, 0.9],
        "blowing_snow_active": [False, False, True, True],
    })
    meta = {"partition_protocol": {"normalization_start_idx": 0, "normalization_end_idx": 4}}

    labels, _, thresholds = module.operating_condition_labels(
        truth,
        meta,
        activity_aligned_transport_demand=True,
    )

    assert thresholds["generator_flux_demand_state"] == pytest.approx(0.5)
    assert thresholds["generator_particle_demand_state"] == pytest.approx(0.5)
    assert labels[:2].tolist() == ["unavailable", "unavailable"]
    assert labels[2] == "flux_0_particle_0_thermal_1"
    assert labels[3] == "flux_1_particle_1_thermal_1"


def test_subset_geometry_uses_same_domain_for_static_opportunity_gap() -> None:
    import pandas as pd

    module = _load_script("109_v32_audit_subset_forecast_geometry.py")
    frame = pd.DataFrame([
        {"candidate": "a", "selected_sensor_ids": "a", "steady_cost": 1.0,
         "state": "unavailable", "oracle_loss": 0.0},
        {"candidate": "b", "selected_sensor_ids": "b", "steady_cost": 1.0,
         "state": "unavailable", "oracle_loss": 10.0},
        {"candidate": "a", "selected_sensor_ids": "a", "steady_cost": 1.0,
         "state": "low", "oracle_loss": 4.0},
        {"candidate": "a", "selected_sensor_ids": "a", "steady_cost": 1.0,
         "state": "high", "oracle_loss": 4.0},
        {"candidate": "b", "selected_sensor_ids": "b", "steady_cost": 1.0,
         "state": "low", "oracle_loss": 1.0},
        {"candidate": "b", "selected_sensor_ids": "b", "steady_cost": 1.0,
         "state": "high", "oracle_loss": 3.0},
    ])

    summary = module.summarize_condition_geometry(
        frame,
        condition_column="state",
        epsilons=[0.01],
    )

    assert summary["best_static_candidate"] == "b"
    assert summary["best_static_loss"] == pytest.approx(2.0)
    assert summary["weighted_best_loss"] == pytest.approx(2.0)
    assert summary["opportunity_gap_best_static_minus_conditionwise"] == pytest.approx(0.0)


def test_receding_diagnostic_propagates_frozen_sensor_quality_config() -> None:
    module = _load_script("99_v32_receding_upper.py")
    command = ["python", "diagnostic.py"]
    metadata = {
        "sensor_quality": {
            "columns": ["quality_a", "quality_b"],
            "max_noise_multiplier": 6.0,
            "availability_floor": 0.2,
        }
    }

    module.append_sensor_quality_args(command, metadata)

    assert command[-7:] == [
        "--sensor-quality-columns",
        "quality_a",
        "quality_b",
        "--sensor-quality-max-noise-multiplier",
        "6.0",
        "--sensor-quality-availability-floor",
        "0.2",
    ]


def test_quality_gate_reconstructs_static_normalized_macro() -> None:
    module = _load_script("103_v32_collect_quality_scene_gate.py")
    import pandas as pd

    static = pd.Series({
        "oracle_loss_subtype_particle": 0.4,
        "oracle_loss_subtype_particle_staticnorm": 0.8,
        "oracle_loss_subtype_flux": 0.6,
        "oracle_loss_subtype_flux_staticnorm": 1.2,
        "oracle_loss_subtype_thermal": 0.5,
        "oracle_loss_subtype_thermal_staticnorm": 1.0,
    })
    candidate = pd.Series({
        "oracle_loss_subtype_particle": 0.25,
        "oracle_loss_subtype_flux": 0.50,
        "oracle_loss_subtype_thermal": 0.75,
    })

    assert module.staticnorm_macro(candidate, static) == 1.0


def test_expected_observability_reads_frozen_validation_static_action(tmp_path: Path) -> None:
    module = _load_script("118_v32_audit_expected_cost_observability.py")
    import pandas as pd

    pd.DataFrame(
        [
            {"action_idx": 6, "oracle_loss_mean": 0.20},
            {"action_idx": 3, "oracle_loss_mean": 0.25},
        ]
    ).to_csv(tmp_path / "validation_static_candidates.csv", index=False)

    assert module.validation_static_index(tmp_path, candidate_count=19) == 6


def test_state_bin_transfer_uses_online_demand_features_and_feasibility() -> None:
    module = _load_script("120_v32_audit_state_bin_transfer.py")
    x = np.zeros((4, 16), dtype=float)
    x[:, 7:10] = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )
    bins = module.bin_ids(x, np.asarray([0.5, 0.5, 0.5]))
    assert bins.tolist() == [0, 2, 5, 7]

    costs = np.asarray([[1.0, 2.0], [2.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
    masks = np.ones_like(costs, dtype=bool)
    table = module.action_cost_table(costs, masks, bins, candidate_count=2)
    metrics = module.evaluate(table, bins, costs, masks, static_index=0)
    assert metrics["lookup_cost_mean"] == 1.0
    assert metrics["static_minus_lookup_cost"] == 0.5
    assert module.finite_ranking(np.asarray([np.inf, 0.3, np.nan, 0.1])) == [3, 1]


def test_state_bin_transfer_falls_back_to_global_action_cost_for_empty_bins() -> None:
    module = _load_script("120_v32_audit_state_bin_transfer.py")
    costs = np.asarray([[0.2, 0.5], [0.4, 0.1]], dtype=float)
    masks = np.asarray([[True, True], [True, False]], dtype=bool)
    bins = np.asarray([0, 0], dtype=int)

    table = module.action_cost_table(costs, masks, bins, candidate_count=2)

    assert np.allclose(table[0], [0.3, 0.5])
    assert np.allclose(table[7], [0.3, 0.5])
